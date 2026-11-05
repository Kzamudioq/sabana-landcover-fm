"""Read the exported rasters, cut chips per split, compute normalisation statistics.

Chips never straddle splits: a chip is kept only if every labelled pixel in it
belongs to one split, so train and test pixels are always >= one 5 km block apart
from each other's chips (spatial cross-validation, no neighbour leakage).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .classes import N_CLASSES, NODATA

S2_NODATA = -32768
BAND_NAMES = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]  # Prithvi/HLS names
SPLITS = {1: "train", 2: "val", 3: "test"}


@dataclass
class Scene:
    image: np.ndarray       # (6, H, W) int16 reflectance x 1e4, S2_NODATA where missing
    worldcover: np.ndarray  # (H, W) uint8 scheme classes, NODATA
    dynamicworld: np.ndarray
    split: np.ndarray       # (H, W) uint8 1/2/3, NODATA
    profile: dict           # rasterio profile of the image (transform, crs)

    @property
    def valid(self) -> np.ndarray:
        return (self.image != S2_NODATA).all(0) & (self.worldcover != NODATA) & (self.split != NODATA)


def load_scene(s2_path: str | Path, labels_path: str | Path) -> Scene:
    import rasterio

    with rasterio.open(s2_path) as src:
        image = src.read()
        profile = src.profile.copy()
    with rasterio.open(labels_path) as src:
        if src.shape != image.shape[1:] or src.transform != profile["transform"]:
            raise ValueError("labels and image are not on the same grid; export both from 01_export_dataset.js")
        wc, dw, split = src.read()
    return Scene(image.astype(np.int16), wc, dw, split, profile)


def chip_windows(h: int, w: int, size: int, stride: int):
    for r in range(0, max(h - size, 0) + 1, stride):
        for c in range(0, max(w - size, 0) + 1, stride):
            yield r, c


def make_chips(scene: Scene, out_dir: str | Path, size: int = 224, stride: int = 224,
               min_valid: float = 0.5) -> dict[str, int]:
    """Write chips as .npz {image int16 (6,s,s), label uint8 (s,s)} under out_dir/{train,val,test}."""
    out = Path(out_dir)
    for name in SPLITS.values():
        (out / name).mkdir(parents=True, exist_ok=True)
    counts = {name: 0 for name in SPLITS.values()}
    valid = scene.valid
    h, w = valid.shape
    for r, c in chip_windows(h, w, size, stride):
        win = (slice(r, r + size), slice(c, c + size))
        v = valid[win]
        if v.mean() < min_valid:
            continue
        sp = np.unique(scene.split[win][v])
        if len(sp) != 1 or int(sp[0]) not in SPLITS:
            continue  # straddles a block border between splits: drop it
        name = SPLITS[int(sp[0])]
        label = np.where(v, scene.worldcover[win], NODATA).astype(np.uint8)
        np.savez_compressed(out / name / f"r{r:05d}_c{c:05d}.npz", image=scene.image[(slice(None),) + win],
                            label=label)
        counts[name] += 1
    return counts


def band_stats(chip_dir: str | Path, max_chips: int = 400, seed: int = 0) -> dict:
    """Per-band mean/std over valid pixels of training chips (used to normalise the model input)."""
    files = sorted(Path(chip_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no chips in {chip_dir}")
    rng = np.random.default_rng(seed)
    files = [files[i] for i in rng.permutation(len(files))[:max_chips]]
    s = np.zeros(6)
    s2 = np.zeros(6)
    n = 0
    for f in files:
        d = np.load(f)
        x = d["image"].reshape(6, -1).astype(np.float64)
        m = (d["label"].reshape(-1) != NODATA) & (x != S2_NODATA).all(0)
        x = x[:, m]
        s += x.sum(1)
        s2 += (x * x).sum(1)
        n += x.shape[1]
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean**2, 1e-6))
    return {"bands": BAND_NAMES, "mean": mean.round(2).tolist(), "std": std.round(2).tolist(), "pixels": int(n)}


def class_frequencies(chip_dir: str | Path) -> np.ndarray:
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    for f in Path(chip_dir).glob("*.npz"):
        lab = np.load(f)["label"]
        counts += np.bincount(lab[lab != NODATA].ravel(), minlength=N_CLASSES)[:N_CLASSES]
    return counts


def class_weights(counts: np.ndarray, power: float = 0.5) -> list[float]:
    """Inverse-frequency^power weights, normalised to mean 1 (rare classes count more in the loss)."""
    f = counts / max(counts.sum(), 1)
    w = np.where(f > 0, (1.0 / np.maximum(f, 1e-9)) ** power, 0.0)
    w = w / w[w > 0].mean()
    return w.round(4).tolist()


def save_json(obj, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))
