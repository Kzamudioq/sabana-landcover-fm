"""A small fake 'Sabana' with the same files the GEE script exports, for tests and dry runs.

Classes are blobs with distinct spectra (+ noise); WorldCover = truth with ~8 % label noise;
Dynamic World = truth with more confusion between grass/crop/shrub; split = 3 x 3 blocks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Rough Sentinel-2 reflectance (x 1e4) for B2 B3 B4 B8A B11 B12 per class.
SPECTRA = np.array([
    [600, 700, 500, 300, 150, 100],       # water
    [300, 550, 300, 3200, 1500, 700],     # trees
    [450, 750, 600, 2600, 2200, 1300],    # shrub
    [500, 850, 650, 3000, 2600, 1600],    # grass
    [600, 900, 900, 2400, 2900, 2100],    # crop
    [1100, 1150, 1250, 1700, 2100, 2000], # built
    [1300, 1500, 1800, 2200, 3200, 3000], # bare
], dtype=float)


def truth_map(h: int, w: int, rng) -> np.ndarray:
    """Smooth random class field: argmax of 7 blurred noise fields."""
    from scipy.ndimage import gaussian_filter

    fields = np.stack([gaussian_filter(rng.normal(size=(h, w)), sigma=9) for _ in range(7)])
    fields[0] -= 0.12  # water and bare rarer
    fields[6] -= 0.08
    return fields.argmax(0).astype(np.uint8)


def write_scene(out: str | Path, h: int = 672, w: int = 672, seed: int = 0) -> dict:
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    truth = truth_map(h, w, rng)
    img = SPECTRA[truth].transpose(2, 0, 1) * rng.normal(1.0, 0.06, size=(1, h, w))
    img += rng.normal(0, 60, size=img.shape)
    img = img.round().astype(np.int16)
    img[:, :12, :40] = -32768  # a cloud hole

    wc = truth.copy()
    flip = rng.random((h, w)) < 0.08
    wc[flip] = rng.integers(0, 7, flip.sum())
    dw = truth.copy()
    confuse = np.isin(truth, [2, 3, 4]) & (rng.random((h, w)) < 0.35)
    dw[confuse] = rng.choice([2, 3, 4], confuse.sum())
    blocks = np.array([[1, 1, 2], [1, 3, 1], [3, 1, 1]], dtype=np.uint8)  # 2 test blocks, 1 val
    rr = np.minimum(np.arange(h) * 3 // h, 2)
    cc = np.minimum(np.arange(w) * 3 // w, 2)
    split = blocks[rr[:, None], cc[None, :]]
    nod = (img == -32768).any(0)
    for a in (wc, dw, split):
        a[nod] = 255

    transform = from_origin(590000, 540000, 10, 10)  # EPSG:32618, Bogotá-ish
    prof = dict(driver="GTiff", height=h, width=w, crs="EPSG:32618", transform=transform)
    with rasterio.open(out / "s2_2021_hls6.tif", "w", count=6, dtype="int16", nodata=-32768, **prof) as dst:
        dst.write(img)
    with rasterio.open(out / "labels_2021.tif", "w", count=3, dtype="uint8", nodata=255, **prof) as dst:
        dst.write(np.stack([wc, dw, split]))

    # Reference sample: stratified by WorldCover inside test blocks, labelled with the TRUTH
    from pyproj import Transformer

    test = (split == 3) & (wc != 255)
    rows = []
    for k in range(7):
        r, c = np.nonzero(test & (wc == k))
        pick = rng.choice(len(r), size=min(40, len(r)), replace=False)
        rows += [(rr, cc, k) for rr, cc in zip(r[pick], c[pick])]
    tf = Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True)
    pts = []
    for i, (r, c, k) in enumerate(rows, 1):
        x, y = rasterio.transform.xy(transform, r, c)
        lon, lat = tf.transform(x, y)
        pts.append({"id": i, "lon": lon, "lat": lat, "stratum": k, "ref_class": int(truth[r, c])})
    df = pd.DataFrame(pts)
    ref = out / "ref"
    ref.mkdir(exist_ok=True)
    df[["id", "lon", "lat"]].to_csv(ref / "ref_points.csv", index=False)
    df[["id", "stratum"]].to_csv(ref / "ref_strata.csv", index=False)
    df[["id", "ref_class"]].to_csv(ref / "ref_labels.csv", index=False)
    counts = np.bincount(wc[test], minlength=7)
    pd.DataFrame({"stratum": range(7), "pixels": counts}).to_csv(ref / "strata_pixels.csv", index=False)
    return {"truth": truth, "split": split, "wc": wc}
