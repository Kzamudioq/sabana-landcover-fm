"""Accuracy and area estimation against photo-interpreted reference points.

The reference sample is stratified by WorldCover class (not by any of the maps being
compared), so the same points can judge every map fairly. With strata different from
the map classes, the textbook Olofsson et al. (2014) formulas do not apply directly;
this module implements the general estimators of Stehman (2014):

  Stehman, S. V. (2014). Estimating area and map accuracy for stratified random sampling
  when the strata are different from the map classes. Int. J. Remote Sensing 35(13).

Every estimate comes with a standard error from the stratified variance formula.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .classes import N_CLASSES, NAMES, NODATA


@dataclass
class Design:
    stratum: np.ndarray      # (n,) stratum of each sample unit
    N_h: dict[int, int]      # population size (pixels) of each stratum

    def __post_init__(self):
        self.strata = sorted(self.N_h)
        self.N = float(sum(self.N_h.values()))


def _total(design: Design, y: np.ndarray) -> tuple[float, float]:
    """Stratified estimate of the population total of y and its variance."""
    t, v = 0.0, 0.0
    for h in design.strata:
        yh = y[design.stratum == h].astype(float)
        n_h, N_h = len(yh), design.N_h[h]
        if n_h == 0:
            continue
        t += N_h * yh.mean()
        if n_h > 1:
            v += N_h**2 * (1 - n_h / N_h) * yh.var(ddof=1) / n_h
    return t, v


def proportion(design: Design, y: np.ndarray) -> tuple[float, float]:
    """Estimated population proportion with y=1 and its standard error."""
    t, v = _total(design, y)
    return t / design.N, np.sqrt(v) / design.N


def ratio(design: Design, y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Ratio estimator R = Y/X (e.g. user's or producer's accuracy) and its standard error."""
    Y, _ = _total(design, y)
    X, _ = _total(design, x)
    if X == 0:
        return float("nan"), float("nan")
    R = Y / X
    v = 0.0
    for h in design.strata:
        sel = design.stratum == h
        yh, xh = y[sel].astype(float), x[sel].astype(float)
        n_h, N_h = len(yh), design.N_h[h]
        if n_h < 2:
            continue
        s2y, s2x = yh.var(ddof=1), xh.var(ddof=1)
        sxy = np.cov(yh, xh, ddof=1)[0, 1]
        v += N_h**2 * (1 - n_h / N_h) * (s2y + R**2 * s2x - 2 * R * sxy) / n_h
    return R, np.sqrt(max(v, 0.0)) / X  # tiny negative values are rounding noise


def assess(design: Design, ref: np.ndarray, mapped: np.ndarray, pixel_area_m2: float = 100.0) -> dict:
    """Full accuracy/area assessment of one map against the reference labels."""
    keep = (ref != NODATA) & (mapped != NODATA)
    d = Design(design.stratum[keep], design.N_h)
    r, m = ref[keep].astype(int), mapped[keep].astype(int)

    oa, oa_se = proportion(d, (r == m).astype(int))
    P = np.zeros((N_CLASSES, N_CLASSES))  # estimated area proportions, rows = map, cols = reference
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            P[i, j] = proportion(d, ((m == i) & (r == j)).astype(int))[0]

    rows = []
    for k in range(N_CLASSES):
        hit = ((m == k) & (r == k)).astype(int)
        ua, ua_se = ratio(d, hit, (m == k).astype(int))
        pa, pa_se = ratio(d, hit, (r == k).astype(int))
        area_p, area_se = proportion(d, (r == k).astype(int))
        f1 = 2 * ua * pa / (ua + pa) if ua + pa > 0 else float("nan")
        denom = P[k, :].sum() + P[:, k].sum() - P[k, k]
        iou = P[k, k] / denom if denom > 0 else float("nan")
        rows.append({
            "class": NAMES[k], "users_acc": ua, "users_se": ua_se, "producers_acc": pa, "producers_se": pa_se,
            "f1": f1, "iou": iou, "ref_points": int((r == k).sum()),
            "area_m2": area_p * d.N * pixel_area_m2, "area_ci95_m2": 1.96 * area_se * d.N * pixel_area_m2,
            "mapped_area_m2": P[k, :].sum() * d.N * pixel_area_m2,
        })
    per_class = pd.DataFrame(rows)
    present = per_class["ref_points"] > 0
    return {
        "n": int(keep.sum()), "overall_acc": oa, "overall_se": oa_se,
        "macro_f1": float(per_class.loc[present, "f1"].mean()),
        "miou": float(per_class.loc[present, "iou"].mean()),
        "per_class": per_class, "error_matrix": P,
    }


def sample_raster(path: str, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Values of a single-band (or band 1) raster at WGS84 points; NODATA outside."""
    import rasterio
    from pyproj import Transformer

    with rasterio.open(path) as src:
        xs, ys = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform(lon, lat)
        vals = np.array([v[0] for v in src.sample(zip(xs, ys))], dtype=np.int64)
        inside = np.array([0 <= r < src.height and 0 <= c < src.width
                           for r, c in (src.index(x, y) for x, y in zip(xs, ys))])
    vals[~inside] = NODATA
    return vals.astype(np.uint8)


def load_reference(points_csv: str, labels_csv: str, strata_csv: str, strata_pixels_csv: str):
    """Join the blind points, the interpreter's labels and the (sealed) strata into one table + design."""
    pts = pd.read_csv(points_csv)
    lab = pd.read_csv(labels_csv)
    strata = pd.read_csv(strata_csv)
    df = pts.merge(lab[["id", "ref_class"]], on="id", how="left").merge(strata, on="id", how="left")
    df["ref_class"] = pd.to_numeric(df["ref_class"], errors="coerce").fillna(NODATA).astype(int)
    df.loc[~df["ref_class"].between(0, N_CLASSES - 1), "ref_class"] = NODATA
    sp = pd.read_csv(strata_pixels_csv)
    design = Design(df["stratum"].to_numpy(), dict(zip(sp["stratum"].astype(int), sp["pixels"].astype(int))))
    return df, design


def agreement_with_labels(pred: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict:
    """Pixel agreement (accuracy, mIoU) with WorldCover in the test blocks. Not accuracy: WorldCover has errors."""
    p, t = pred[mask].astype(int), labels[mask].astype(int)
    ok = (p != NODATA) & (t != NODATA)
    p, t = p[ok], t[ok]
    cm = np.bincount(t * N_CLASSES + p, minlength=N_CLASSES**2).reshape(N_CLASSES, N_CLASSES)
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return {"pixel_acc": float(inter.sum() / max(cm.sum(), 1)), "miou": float(np.nanmean(iou)),
            "iou": iou.tolist()}
