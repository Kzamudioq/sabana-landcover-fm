"""Pixel baselines on the same composite and the same training blocks as the model.

- K-Means: the unsupervised method of the original Cerros analysis, made comparable.
  It is fitted ONCE (not per year) and every cluster is named after the WorldCover
  class it overlaps most in the training blocks, so cluster ids mean the same thing everywhere.
- Random Forest: the classic supervised pixel classifier, with spectral indices added.
"""
from __future__ import annotations

import numpy as np

from .classes import N_CLASSES, NODATA
from .data import S2_NODATA, Scene


def features(image: np.ndarray) -> np.ndarray:
    """(6, ...) reflectance -> (..., 9): six bands + NDVI, NDWI (McFeeters), NDBI."""
    x = image.astype(np.float32) / 10000.0
    b, g, r, nir, sw1, _ = x

    def nd(a, c):
        return (a - c) / np.maximum(a + c, 1e-6)

    f = np.concatenate([x, nd(nir, r)[None], nd(g, nir)[None], nd(sw1, nir)[None]], axis=0)
    return np.moveaxis(f, 0, -1)


def sample_training_pixels(scene: Scene, n_per_class: int = 4000, seed: int = 0):
    """Balanced random pixels from the training blocks: X (n, 9), y (n,)."""
    rng = np.random.default_rng(seed)
    ok = scene.valid & (scene.split == 1)
    rows, cols, ys = [], [], []
    for k in range(N_CLASSES):
        r, c = np.nonzero(ok & (scene.worldcover == k))
        if len(r) == 0:
            continue
        pick = rng.choice(len(r), size=min(n_per_class, len(r)), replace=False)
        rows.append(r[pick])
        cols.append(c[pick])
        ys.append(np.full(len(pick), k, dtype=np.uint8))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    X = features(scene.image[:, rows, cols])
    return X, np.concatenate(ys)


def _predict_blocks(scene: Scene, fn, block: int = 1024) -> np.ndarray:
    _, h, w = scene.image.shape
    out = np.full((h, w), NODATA, dtype=np.uint8)
    for r in range(0, h, block):
        for c in range(0, w, block):
            img = scene.image[:, r:r + block, c:c + block]
            ok = (img != S2_NODATA).all(0)
            if not ok.any():
                continue
            f = features(img)[ok]
            tile = out[r:r + block, c:c + block]
            tile[ok] = fn(f)
    return out


class KMeansBaseline:
    def __init__(self, k: int = 12, seed: int = 0):
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.preprocessing import StandardScaler

        self.k = k
        self.scaler = StandardScaler()
        self.km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=5, batch_size=8192)
        self.cluster_to_class = np.zeros(k, dtype=np.uint8)

    def fit(self, scene: Scene, n_pixels: int = 200_000, seed: int = 0) -> "KMeansBaseline":
        rng = np.random.default_rng(seed)
        r, c = np.nonzero(scene.valid & (scene.split == 1))
        pick = rng.choice(len(r), size=min(n_pixels, len(r)), replace=False)
        X = features(scene.image[:, r[pick], c[pick]])
        y = scene.worldcover[r[pick], c[pick]]
        labels = self.km.fit_predict(self.scaler.fit_transform(X))
        for j in range(self.k):  # name each cluster after its majority WorldCover class
            members = y[labels == j]
            self.cluster_to_class[j] = np.bincount(members, minlength=N_CLASSES).argmax() if len(members) else 3
        return self

    def predict_pixels(self, X: np.ndarray) -> np.ndarray:
        return self.cluster_to_class[self.km.predict(self.scaler.transform(X))]

    def predict(self, scene: Scene) -> np.ndarray:
        return _predict_blocks(scene, self.predict_pixels)


class RandomForestBaseline:
    def __init__(self, n_estimators: int = 300, seed: int = 0):
        from sklearn.ensemble import RandomForestClassifier

        self.rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=2, n_jobs=-1,
                                         random_state=seed)

    def fit(self, scene: Scene, n_per_class: int = 4000, seed: int = 0) -> "RandomForestBaseline":
        X, y = sample_training_pixels(scene, n_per_class, seed)
        self.rf.fit(X, y)
        return self

    def predict_pixels(self, X: np.ndarray) -> np.ndarray:
        return self.rf.predict(X).astype(np.uint8)

    def predict(self, scene: Scene) -> np.ndarray:
        return _predict_blocks(scene, self.predict_pixels)
