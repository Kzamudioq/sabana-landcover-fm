"""End to end on a synthetic scene: prepare -> baselines -> Prithvi (tiny, random init) -> evaluate."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sabana_lc.cli import main
from sabana_lc.data import load_scene

from synthetic import write_scene


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("sabana")
    info = write_scene(d)
    work = d / "work"
    main(["prepare", "--s2", str(d / "s2_2021_hls6.tif"), "--labels", str(d / "labels_2021.tif"),
          "--out", str(work), "--chip", "64", "--stride", "64"])
    return d, work, info


def test_chips_never_mix_splits(run):
    d, work, info = run
    scene = load_scene(d / "s2_2021_hls6.tif", d / "labels_2021.tif")
    for split, name in ((1, "train"), (2, "val"), (3, "test")):
        files = list((work / "chips" / name).glob("*.npz"))
        assert files, name
        for f in files:
            r, c = (int(p[1:]) for p in f.stem.split("_"))
            sp = scene.split[r:r + 64, c:c + 64]
            assert set(np.unique(sp[sp != 255])) == {split}
    stats = json.loads((work / "stats.json").read_text())
    assert len(stats["mean"]) == 6 and all(s > 0 for s in stats["std"])
    assert len(stats["class_weights"]) == 7


def test_baselines_and_model_and_evaluation(run):
    d, work, info = run
    main(["baselines", "--work", str(work), "--k", "10"])
    main(["train", "--work", str(work), "--backbone", "prithvi_eo_tiny", "--no-pretrained", "--epochs", "2",
          "--batch", "4", "--workers", "0", "--lr", "5e-4"])
    ckpt = work / "runs" / "prithvi" / "best.ckpt"
    assert ckpt.exists()
    main(["predict", "--work", str(work), "--ckpt", str(ckpt)])
    for m in ("kmeans", "random_forest", "dynamicworld", "prithvi"):
        assert (work / "maps" / f"{m}.tif").exists()

    main(["evaluate", "--work", str(work), "--ref", str(d / "ref")])
    s = pd.read_csv(work / "results" / "summary.csv").set_index("map")
    # Synthetic spectra are separable: RF must be near perfect against the TRUE reference labels,
    # and clearly better than its agreement with the noisy WorldCover labels it was trained on.
    assert s.loc["random_forest", "overall_acc"] > 0.9
    assert s.loc["random_forest", "overall_acc"] > s.loc["random_forest", "agreement_worldcover"]
    assert s.loc["kmeans", "overall_acc"] > 0.5
    assert s.loc["dynamicworld", "overall_acc"] < s.loc["random_forest", "overall_acc"]
    assert 0 <= s.loc["prithvi", "overall_acc"] <= 1  # 2 epochs, random init: only checks the plumbing
    assert (work / "results" / "f1_by_class.png").exists()
    assert "ha +" in (work / "results" / "results.md").read_text()
