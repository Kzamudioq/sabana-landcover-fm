"""Mapa de coberturas de un ráster clasificado (0-6, 255 = sin dato), con grilla, escala, norte y leyenda interna.

python scripts/mapa_coberturas.py --mapa resultados/prithvi_2021.tif --salida docs/img/prithvi_2021.png \
       --titulo "Coberturas 2021 · Prithvi-EO-2.0 ajustado"
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
import matplotlib.patheffects as pe
from rasterio.enums import Resampling

from sabana_lc.classes import COLORS, LABELS_ES, N_CLASSES, NODATA

TINTA = "#1d1d1b"
GRIS = "#5f5e5a"
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]   # texto legible sin caja de fondo


def ha_m2(m2: float) -> str:
    """Área como 'X ha + NNNN m²' (separador de miles con punto)."""
    m2 = int(round(m2))
    return f"{m2 // 10_000:,} ha + {m2 % 10_000:04d} m²".replace(",", ".")


def escala(ax, x0, y0, largo, alto, tramos=4):
    paso = largo / tramos
    for i in range(tramos):
        ax.add_patch(Rectangle((x0 + i * paso, y0), paso, alto, facecolor=TINTA if i % 2 == 0 else "white",
                               edgecolor=TINTA, linewidth=0.8, zorder=6))
    for i in (0, tramos // 2, tramos):
        ax.text(x0 + i * paso, y0 + alto * 1.5, f"{i * paso / 1000:.0f}" + (" km" if i == tramos else ""),
                ha="center", va="bottom", fontsize=8, color=TINTA, zorder=6, path_effects=HALO)


def norte(ax, x, y, alto):
    w = alto * 0.36
    ax.add_patch(Polygon([(x, y + alto), (x - w / 2, y), (x, y + alto * 0.28)], closed=True, facecolor=TINTA,
                         edgecolor=TINTA, linewidth=0.8, zorder=6))
    ax.add_patch(Polygon([(x, y + alto), (x + w / 2, y), (x, y + alto * 0.28)], closed=True, facecolor="white",
                         edgecolor=TINTA, linewidth=0.8, zorder=6))
    ax.text(x, y + alto * 1.08, "N", ha="center", va="bottom", fontsize=10, color=TINTA, zorder=6,
            path_effects=HALO)


def grilla(ax, paso):
    fmt = mticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", "."))
    for eje in (ax.xaxis, ax.yaxis):
        eje.set_major_locator(mticker.MultipleLocator(paso))
        eje.set_major_formatter(fmt)
    ax.grid(True, color=GRIS, linewidth=0.45, alpha=0.6, linestyle=(0, (4, 3)), zorder=4)
    ax.tick_params(labelsize=6.5, colors="#52514e", length=3, direction="out", top=True, right=True,
                   labeltop=True, labelright=False)
    plt.setp(ax.get_yticklabels(), rotation=90, va="center")
    for sp in ax.spines.values():
        sp.set_color("#52514e")
        sp.set_linewidth(0.8)


def leyenda(ax, conteos, pix_m2):
    """Caja interna compacta arriba a la izquierda: color, clase, porcentaje y área en 'ha + m²'."""
    total = conteos.sum()
    x0, dy, ancho = 0.02, 0.026, 0.355
    y0 = 0.985 - 0.042 - dy * N_CLASSES       # filas desde arriba
    ax.add_patch(FancyBboxPatch((x0 - 0.008, y0 - 0.01), ancho, dy * N_CLASSES + 0.042, transform=ax.transAxes,
                                boxstyle="round,pad=0,rounding_size=0.006", facecolor="white", alpha=0.92,
                                edgecolor="none", zorder=5))
    ax.text(x0, y0 + dy * N_CLASSES + 0.004, "Cobertura · % · área", transform=ax.transAxes,
            fontsize=7.3, fontweight="bold", color=TINTA, zorder=6)
    for k in range(N_CLASSES):
        y = y0 + dy * (N_CLASSES - 1 - k)
        ax.add_patch(Rectangle((x0, y), 0.02, 0.017, transform=ax.transAxes, facecolor=COLORS[k],
                               edgecolor="white", linewidth=0.5, zorder=6))
        ax.text(x0 + 0.026, y + 0.0085, LABELS_ES[k], transform=ax.transAxes, fontsize=6.6, va="center",
                color=TINTA, zorder=6)
        ax.text(x0 + 0.178, y + 0.0085, f"{conteos[k] / total * 100:4.1f} %".replace(".", ","),
                transform=ax.transAxes, fontsize=6.6, va="center", ha="right", color=TINTA, zorder=6)
        ax.text(x0 + 0.34, y + 0.0085, ha_m2(conteos[k] * pix_m2), transform=ax.transAxes, fontsize=6.6,
                va="center", ha="right", color="#52514e", zorder=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapa", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--titulo", default="Coberturas 2021")
    ap.add_argument("--fuente", default="Sentinel-2 L2A, mediana 2021 · Clasificación: Prithvi-EO-2.0 ajustado "
                                        "con WorldCover 2021 · Áreas contadas en el mapa (sin ajuste por exactitud)")
    ap.add_argument("--reduccion", type=int, default=3, help="factor de reducción para dibujar (moda)")
    ap.add_argument("--paso-grilla", type=float, default=10_000)
    a = ap.parse_args()

    with rasterio.open(a.mapa) as src:
        cl = src.read(1)
        b, crs, res = src.bounds, src.crs, abs(src.res[0])
        vista = src.read(1, out_shape=(src.height // a.reduccion, src.width // a.reduccion),
                         resampling=Resampling.mode)
    conteos = np.bincount(cl[cl != NODATA].ravel(), minlength=N_CLASSES)[:N_CLASSES]

    cmap = ListedColormap(COLORS)
    img = np.ma.masked_equal(vista, NODATA)
    ancho, alto = b.right - b.left, b.top - b.bottom
    fig, ax = plt.subplots(figsize=(9.5, 9.5 * alto / ancho))
    ax.imshow(img, cmap=cmap, vmin=0, vmax=N_CLASSES - 1, extent=(b.left, b.right, b.bottom, b.top),
              interpolation="nearest", zorder=1)
    ax.set_xlim(b.left, b.right)
    ax.set_ylim(b.bottom, b.top)
    grilla(ax, a.paso_grilla)
    leyenda(ax, conteos, res * res)
    escala(ax, b.right - 0.08 * ancho - 10_000, b.bottom + 0.035 * alto, 10_000, ancho * 0.009)
    norte(ax, b.right - 0.06 * ancho, b.top - 0.11 * alto, ancho * 0.05)
    ax.set_title(a.titulo, loc="left", fontsize=13, pad=22, fontweight="bold", color=TINTA)
    epsg = crs.to_epsg()
    ax.text(0.0, -0.04, f"{a.fuente}\nCoordenadas WGS 84 / UTM 18N (EPSG:{epsg}), grilla cada "
            f"{a.paso_grilla / 1000:.0f} km · Área total clasificada: {ha_m2(conteos.sum() * res * res)}",
            transform=ax.transAxes, fontsize=6.3, color="#52514e", va="top")
    fig.savefig(a.salida, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
