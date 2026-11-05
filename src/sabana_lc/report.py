"""Figures for the README: per-class F1 by method, and maps side by side."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .classes import COLORS, LABELS_ES, NAMES, NODATA

BG, FG, MUTED, GRID = "#f6f4ef", "#1b1f1d", "#6b726e", "#d9d6cd"
METHOD_COLORS = {"kmeans": "#b8a99a", "random_forest": "#9bbf4a", "dynamicworld": "#3a7bd5", "prithvi": "#d7263d"}


def _style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": FG, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
                         "figure.facecolor": BG, "axes.facecolor": BG})
    return plt


def figures(work: Path, summary, per_class) -> None:
    plt = _style()
    out = work / "results"
    methods = list(summary["map"])
    fig, ax = plt.subplots(figsize=(11, 4.4))
    x = np.arange(len(NAMES))
    w = 0.8 / max(len(methods), 1)
    for i, m in enumerate(methods):
        f1 = per_class[per_class["map"] == m].set_index("class").reindex(NAMES)["f1"].to_numpy()
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, f1, w, label=m, color=METHOD_COLORS.get(m, "#555"))
    ax.set_xticks(x, LABELS_ES, rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 vs reference points")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, ncol=len(methods), loc="upper left", bbox_to_anchor=(0, 1.12))
    fig.savefig(out / "f1_by_class.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    map_panels(work, methods, out / "maps_panel.png")


def map_panels(work: Path, methods: list[str], path: Path, window: tuple[int, int, int, int] | None = None) -> None:
    import rasterio
    from matplotlib.colors import ListedColormap

    plt = _style()
    cmap = ListedColormap(COLORS)
    fig, axes = plt.subplots(1, len(methods), figsize=(3.6 * len(methods), 4.2))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, methods):
        with rasterio.open(work / "maps" / f"{m}.tif") as src:
            a = src.read(1, window=rasterio.windows.Window(*window) if window else None).astype(float)
        a[a == NODATA] = np.nan
        ax.imshow(a, cmap=cmap, vmin=-0.5, vmax=len(COLORS) - 0.5, interpolation="nearest")
        ax.set_title(m, loc="left", color=FG)
        ax.set_axis_off()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
