"""Command line for the whole experiment.

  python -m sabana_lc prepare   --s2 s2_2021_hls6.tif --labels labels_2021.tif --out work
  python -m sabana_lc baselines --work work
  python -m sabana_lc train     --work work [--backbone prithvi_eo_v2_300_tl --epochs 40]
  python -m sabana_lc predict   --work work --ckpt work/runs/best.ckpt
  python -m sabana_lc evaluate  --work work --ref ref/   (ref_points.csv, ref_labels.csv, ref_strata.csv, strata_pixels.csv)

`work/` keeps the scene paths, chips, normalisation stats, maps and results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .classes import COLORS, NAMES, NODATA, fmt_area


def _state(work: Path) -> dict:
    return json.loads((work / "state.json").read_text())


def _scene(work: Path):
    from .data import load_scene

    st = _state(work)
    return load_scene(st["s2"], st["labels"])


def write_map(pred: np.ndarray, profile: dict, path: Path) -> None:
    import rasterio

    prof = {**profile, "count": 1, "dtype": "uint8", "nodata": NODATA, "compress": "deflate",
            "tiled": True, "blockxsize": 256, "blockysize": 256}
    prof.pop("photometric", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(pred, 1)
        dst.write_colormap(1, {i: tuple(int(c[k:k + 2], 16) for k in (1, 3, 5)) + (255,)
                               for i, c in enumerate(COLORS)})


# ------------------------------------------------------------------ commands
def cmd_prepare(a) -> None:
    from .data import band_stats, class_frequencies, class_weights, load_scene, make_chips, save_json

    work = Path(a.out)
    work.mkdir(parents=True, exist_ok=True)
    save_json({"s2": str(Path(a.s2).resolve()), "labels": str(Path(a.labels).resolve())}, work / "state.json")
    scene = load_scene(a.s2, a.labels)
    counts = make_chips(scene, work / "chips", size=a.chip, stride=a.stride)
    stats = band_stats(work / "chips" / "train")
    freq = class_frequencies(work / "chips" / "train")
    stats["class_pixels"] = freq.tolist()
    stats["class_weights"] = class_weights(freq)
    save_json(stats, work / "stats.json")
    write_map(scene.dynamicworld, scene.profile, work / "maps" / "dynamicworld.tif")
    print(f"chips {counts}")
    print("train pixels per class:", dict(zip(NAMES, freq.tolist())))


def cmd_baselines(a) -> None:
    import joblib

    from .baselines import KMeansBaseline, RandomForestBaseline

    work = Path(a.work)
    scene = _scene(work)
    for name, model in (("kmeans", KMeansBaseline(k=a.k)), ("random_forest", RandomForestBaseline())):
        model.fit(scene)
        write_map(model.predict(scene), scene.profile, work / "maps" / f"{name}.tif")
        joblib.dump(model, work / f"{name}.joblib")
        print(f"{name} -> {work / 'maps' / (name + '.tif')}")


def _cfg(work: Path, a):
    from .model import TrainConfig

    stats = json.loads((work / "stats.json").read_text())
    return TrainConfig(backbone=a.backbone, pretrained=not a.no_pretrained, freeze_backbone=a.freeze,
                       epochs=a.epochs, batch_size=a.batch, lr=a.lr, mean=stats["mean"], std=stats["std"],
                       class_weights=stats["class_weights"], num_workers=a.workers)


def cmd_train(a) -> None:
    from .model import train

    work = Path(a.work)
    _, best = train(work / "chips", work / "runs" / a.name, _cfg(work, a),
                    limit_batches=a.limit_batches)
    print(f"best checkpoint: {best}")


def cmd_predict(a) -> None:
    from dataclasses import fields

    from .model import TrainConfig, load_model, predict_scene

    work = Path(a.work)
    saved = json.loads((Path(a.ckpt).parent / "train_config.json").read_text())
    cfg = TrainConfig(**{k: v for k, v in saved.items() if k in {f.name for f in fields(TrainConfig)}})
    scene = _scene(work)
    pred = predict_scene(load_model(a.ckpt, cfg), scene, cfg.mean, cfg.std, chip=cfg.chip_size)
    out = work / "maps" / f"{a.name}.tif"
    write_map(pred, scene.profile, out)
    print(f"-> {out}")


def cmd_evaluate(a) -> None:
    import pandas as pd

    from .evaluate import agreement_with_labels, assess, load_reference, sample_raster
    from .report import figures

    work = Path(a.work)
    ref = Path(a.ref)
    df, design = load_reference(ref / "ref_points.csv", ref / "ref_labels.csv", ref / "ref_strata.csv",
                                ref / "strata_pixels.csv")
    scene = _scene(work)
    test = scene.valid & (scene.split == 3)
    out = work / "results"
    out.mkdir(exist_ok=True)
    summary, per_class, mats = [], [], {}
    maps = sorted((work / "maps").glob("*.tif"))
    for path in maps:
        import rasterio

        name = path.stem
        mapped = sample_raster(str(path), df["lon"].to_numpy(), df["lat"].to_numpy())
        res = assess(design, df["ref_class"].to_numpy(), mapped)
        with rasterio.open(path) as src:
            agree = agreement_with_labels(src.read(1), scene.worldcover, test)
        summary.append({"map": name, "points": res["n"], "overall_acc": res["overall_acc"],
                        "overall_se": res["overall_se"], "macro_f1": res["macro_f1"], "miou_ref": res["miou"],
                        "agreement_worldcover": agree["pixel_acc"], "miou_worldcover": agree["miou"]})
        pc = res["per_class"].assign(map=name)
        per_class.append(pc)
        mats[name] = res["error_matrix"].tolist()
    summary = pd.DataFrame(summary).sort_values("macro_f1", ascending=False)
    per_class = pd.concat(per_class)
    summary.to_csv(out / "summary.csv", index=False)
    per_class.to_csv(out / "per_class.csv", index=False)
    (out / "error_matrices.json").write_text(json.dumps(mats))

    lines = ["| Map | OA ± 95% CI | Macro F1 | mIoU | Agreement with WorldCover |", "|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r['map']} | {r['overall_acc']:.3f} ± {1.96 * r['overall_se']:.3f} | {r['macro_f1']:.3f} "
                     f"| {r['miou_ref']:.3f} | {r['agreement_worldcover']:.3f} |")
    best = summary.iloc[0]["map"]
    lines += ["", f"Estimated area by class (reference, test blocks), from the sample — best map: {best}", "",
              "| Class | Area | 95% CI | Mapped by best map |", "|---|---|---|---|"]
    for _, r in per_class[per_class["map"] == best].iterrows():
        lines.append(f"| {r['class']} | {fmt_area(r['area_m2'])} | ± {fmt_area(r['area_ci95_m2'])} "
                     f"| {fmt_area(r['mapped_area_m2'])} |")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    figures(work, summary, per_class)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sabana_lc", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--s2", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chip", type=int, default=224)
    p.add_argument("--stride", type=int, default=224)
    p = sub.add_parser("baselines")
    p.add_argument("--work", required=True)
    p.add_argument("--k", type=int, default=12)
    for name in ("train", "predict"):
        p = sub.add_parser(name)
        p.add_argument("--work", required=True)
        p.add_argument("--name", default="prithvi")
        if name == "predict":
            p.add_argument("--ckpt", required=True)
            continue
        p.add_argument("--backbone", default="prithvi_eo_v2_300_tl")
        p.add_argument("--no-pretrained", action="store_true")
        p.add_argument("--freeze", action="store_true", help="train only the decoder")
        p.add_argument("--epochs", type=int, default=40)
        p.add_argument("--batch", type=int, default=8)
        p.add_argument("--lr", type=float, default=1e-4)
        p.add_argument("--workers", type=int, default=2)
        p.add_argument("--limit-batches", type=float, default=None)
    p = sub.add_parser("evaluate")
    p.add_argument("--work", required=True)
    p.add_argument("--ref", required=True)
    a = ap.parse_args(argv)
    {"prepare": cmd_prepare, "baselines": cmd_baselines, "train": cmd_train, "predict": cmd_predict,
     "evaluate": cmd_evaluate}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
