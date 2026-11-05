"""The 7-class legend shared by every map in the comparison, and area formatting."""
from __future__ import annotations

import numpy as np

NODATA = 255

CLASSES = [
    # id, short name, Spanish label, colour
    (0, "water", "Agua", "#3a7bd5"),
    (1, "trees", "Bosque / árboles", "#1f6e43"),
    (2, "shrub", "Arbustal", "#9bbf4a"),
    (3, "grass", "Pastos / herbazal", "#d8e27a"),
    (4, "crop", "Cultivos", "#f2b134"),
    (5, "built", "Construido", "#d7263d"),
    (6, "bare", "Suelo desnudo", "#b8a99a"),
]
N_CLASSES = len(CLASSES)
NAMES = [c[1] for c in CLASSES]
LABELS_ES = [c[2] for c in CLASSES]
COLORS = [c[3] for c in CLASSES]

# ESA WorldCover v200 code -> scheme (also done in GEE; repeated here so raw rasters can be used too)
WORLDCOVER = {10: 1, 20: 2, 30: 3, 40: 4, 50: 5, 60: 6, 70: NODATA, 80: 0, 90: 3, 95: 1, 100: 3}
# Dynamic World label index -> scheme
DYNAMIC_WORLD = {0: 0, 1: 1, 2: 3, 3: 3, 4: 4, 5: 2, 6: 5, 7: 6, 8: NODATA}


def remap(arr: np.ndarray, table: dict[int, int], default: int = NODATA) -> np.ndarray:
    lut = np.full(256, default, dtype=np.uint8)
    for k, v in table.items():
        lut[k] = v
    return lut[arr.astype(np.uint8)]


def fmt_area(m2: float) -> str:
    """Area as 'X ha + NNNN m²' (m² part zero-padded to 4 digits, no decimals)."""
    total = int(round(m2))
    sign = "-" if total < 0 else ""
    ha, rest = divmod(abs(total), 10_000)
    return f"{sign}{ha:,} ha + {rest:04d} m²".replace(",", ".")
