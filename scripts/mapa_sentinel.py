"""Imagen de referencia Sentinel-2 2021: color natural y falso color, con grilla, escala y norte.

Además guarda una copia liviana en color natural (8 bits, 20 m) que sí cabe en el repositorio.

python scripts/mapa_sentinel.py --s2 s2_2021_hls6.tif --salida docs/img/sentinel2_2021.png \
       --liviana resultados/s2_2021_color_natural_20m.tif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).parent))
from mapa_coberturas import TINTA, escala, grilla, norte  # noqa: E402

NODATA_S2 = -32768
BANDAS = ["B2", "B3", "B4", "B8A", "B11", "B12"]


def estirar(bandas: np.ndarray, valido: np.ndarray, p=(2, 98)) -> np.ndarray:
    """Estiramiento por percentiles, banda por banda, solo sobre píxeles válidos -> uint8."""
    out = np.zeros(bandas.shape, np.uint8)
    for i, b in enumerate(bandas):
        lo, hi = np.percentile(b[valido], p)
        out[i] = (np.clip((b - lo) / (hi - lo), 0, 1) * 254 + 1).astype(np.uint8)
    out[:, ~valido] = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2", required=True)
    ap.add_argument("--salida", default="docs/img/sentinel2_2021.png")
    ap.add_argument("--liviana", default="resultados/s2_2021_color_natural_20m.tif")
    ap.add_argument("--reduccion", type=int, default=3)
    a = ap.parse_args()

    with rasterio.open(a.s2) as src:
        b, crs = src.bounds, src.crs
        r = a.reduccion
        img = src.read(out_shape=(src.count, src.height // r, src.width // r), resampling=Resampling.average)
        liv = src.read((3, 2, 1), out_shape=(3, src.height // 2, src.width // 2), resampling=Resampling.average)
        t_liv = src.transform * src.transform.scale(src.width / liv.shape[2], src.height / liv.shape[1])

    def combinar(arr, nombres):
        idx = [BANDAS.index(n) for n in nombres]
        sel = arr[idx].astype(float)
        valido = (arr > NODATA_S2 // 2).all(0) & (sel > 0).any(0)
        return estirar(sel, valido).transpose(1, 2, 0), valido

    natural, v = combinar(img, ["B4", "B3", "B2"])
    falso, _ = combinar(img, ["B8A", "B11", "B4"])

    ancho, alto = b.right - b.left, b.top - b.bottom
    fig, axs = plt.subplots(1, 2, figsize=(14, 7.6 * alto / ancho), gridspec_kw={"wspace": 0.08})
    for ax, rgb, sub in ((axs[0], natural, "Color natural (B4-B3-B2)"),
                         (axs[1], falso, "Falso color (B8A-B11-B4): vegetación en naranja, construido en azul, agua oscura")):
        rgba = np.dstack([rgb, np.where(v, 255, 0).astype(np.uint8)])
        ax.imshow(rgba, extent=(b.left, b.right, b.bottom, b.top), interpolation="lanczos")
        ax.set_xlim(b.left, b.right)
        ax.set_ylim(b.bottom, b.top)
        grilla(ax, 10_000)
        escala(ax, b.right - 0.08 * ancho - 10_000, b.bottom + 0.035 * alto, 10_000, ancho * 0.009)
        norte(ax, b.right - 0.06 * ancho, b.top - 0.11 * alto, ancho * 0.05)
        ax.set_title(sub, loc="left", fontsize=10, color=TINTA, pad=18)
    fig.suptitle("Imagen de referencia · Sentinel-2 2021, Sabana de Bogotá", x=0.125, ha="left", y=0.93,
                 fontsize=13, fontweight="bold", color=TINTA)
    axs[0].text(0.0, -0.06, "Copernicus Sentinel-2 L2A (2021), mediana anual con nubes enmascaradas por "
                "Cloud Score+ · 10 m · Coordenadas WGS 84 / UTM 18N (EPSG:32618), grilla cada 10 km",
                transform=axs[0].transAxes, fontsize=6.5, color="#52514e", va="top")
    fig.savefig(a.salida, dpi=150, bbox_inches="tight", facecolor="white")

    valido_l = (liv > NODATA_S2 // 2).all(0)
    rgb_l = estirar(liv.astype(float), valido_l)
    Path(a.liviana).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(a.liviana, "w", driver="GTiff", width=rgb_l.shape[2], height=rgb_l.shape[1], count=3,
                       dtype="uint8", crs=crs, transform=t_liv, nodata=0, compress="DEFLATE", predictor=2,
                       tiled=True) as d:
        d.write(rgb_l)
        d.descriptions = ("Rojo (B4)", "Verde (B3)", "Azul (B2)")


if __name__ == "__main__":
    main()
