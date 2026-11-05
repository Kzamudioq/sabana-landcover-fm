"""Fine-tune Prithvi-EO-2.0 (NASA/IBM geospatial foundation model) for the 7-class map, with TerraTorch.

The encoder is the pretrained ViT; a small UNet decoder turns its tokens back into a
10 m map. Everything else (data, loss, sliding-window inference) is plain PyTorch/Lightning
so the same code runs on a laptop CPU with the 3.5 M "tiny" backbone (tests) and on a
Colab GPU with the 300 M backbone (the real run).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .classes import N_CLASSES, NODATA
from .data import BAND_NAMES, S2_NODATA, Scene

# Encoder blocks fed to the decoder, per backbone depth.
NECK_INDICES = {
    "prithvi_eo_tiny": [0, 1, 2, 3],
    "prithvi_eo_v2_100_tl": [2, 5, 8, 11],
    "prithvi_eo_v2_300_tl": [5, 11, 17, 23],
    "prithvi_eo_v2_300": [5, 11, 17, 23],
    "prithvi_eo_v2_600_tl": [7, 15, 23, 31],
}


@dataclass
class TrainConfig:
    backbone: str = "prithvi_eo_v2_300_tl"
    pretrained: bool = True
    freeze_backbone: bool = False
    chip_size: int = 224
    batch_size: int = 8
    epochs: int = 40
    lr: float = 1e-4
    weight_decay: float = 0.05
    num_workers: int = 2
    class_weights: list[float] | None = None
    mean: list[float] = field(default_factory=lambda: [0.0] * 6)
    std: list[float] = field(default_factory=lambda: [1.0] * 6)
    precision: str = "16-mixed"
    seed: int = 42


# ---------------------------------------------------------------- data
class ChipDataset(Dataset):
    def __init__(self, chip_dir: str | Path, mean, std, augment: bool = False):
        self.files = sorted(Path(chip_dir).glob("*.npz"))
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        x, y = normalise(d["image"], self.mean, self.std), torch.from_numpy(d["label"].astype(np.int64))
        if self.augment:
            k = int(torch.randint(0, 4, ()))
            x, y = torch.rot90(x, k, (1, 2)), torch.rot90(y, k, (0, 1))
            if torch.rand(()) < 0.5:
                x, y = torch.flip(x, (2,)), torch.flip(y, (1,))
        return {"image": x, "mask": y}


def normalise(image: np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    x = torch.from_numpy(image.astype(np.float32))
    nod = (x == S2_NODATA).any(0, keepdim=True)
    x = (x - mean) / std
    return torch.where(nod, torch.zeros_like(x), x)


def make_datamodule(chip_root: str | Path, cfg: TrainConfig):
    import lightning.pytorch as pl

    class DM(pl.LightningDataModule):
        def setup(self, stage=None):
            root = Path(chip_root)
            self.train_ds = ChipDataset(root / "train", cfg.mean, cfg.std, augment=True)
            self.val_ds = ChipDataset(root / "val", cfg.mean, cfg.std)

        def train_dataloader(self):
            return DataLoader(self.train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                              drop_last=len(self.train_ds) > cfg.batch_size)

        def val_dataloader(self):
            return DataLoader(self.val_ds, cfg.batch_size, num_workers=cfg.num_workers)

    return DM()


# ---------------------------------------------------------------- model
def build_task(cfg: TrainConfig):
    from terratorch.tasks import SemanticSegmentationTask

    return SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory",
        model_args=dict(
            backbone=cfg.backbone,
            backbone_pretrained=cfg.pretrained,
            backbone_bands=BAND_NAMES,
            backbone_num_frames=1,
            necks=[{"name": "SelectIndices", "indices": NECK_INDICES[cfg.backbone]},
                   {"name": "ReshapeTokensToImage"},
                   {"name": "LearnedInterpolateToPyramidal"}],
            decoder="UNetDecoder",
            decoder_channels=[512, 256, 128, 64],
            head_dropout=0.1,
            num_classes=N_CLASSES,
        ),
        loss="ce",
        class_weights=cfg.class_weights,
        ignore_index=NODATA,
        lr=cfg.lr,
        optimizer="AdamW",
        optimizer_hparams={"weight_decay": cfg.weight_decay},
        scheduler="CosineAnnealingLR",
        scheduler_hparams={"T_max": cfg.epochs},
        freeze_backbone=cfg.freeze_backbone,
        class_names=[str(i) for i in range(N_CLASSES)],
    )


def train(chip_root: str | Path, out_dir: str | Path, cfg: TrainConfig, accelerator: str = "auto",
          limit_batches: float | int | None = None):
    """Fit and return (task, path to best checkpoint by validation mIoU)."""
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint

    pl.seed_everything(cfg.seed, workers=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(__import__("json").dumps(asdict(cfg), indent=2))
    ckpt = ModelCheckpoint(dirpath=out, filename="best", monitor="val/mIoU", mode="max", save_top_k=1)
    trainer = pl.Trainer(
        max_epochs=cfg.epochs, accelerator=accelerator, devices=1, default_root_dir=out,
        precision=cfg.precision if torch.cuda.is_available() else "32-true",
        callbacks=[ckpt], log_every_n_steps=5, enable_progress_bar=True,
        limit_train_batches=limit_batches, limit_val_batches=limit_batches,
    )
    task = build_task(cfg)
    trainer.fit(task, datamodule=make_datamodule(chip_root, cfg))
    return task, ckpt.best_model_path or None


# ---------------------------------------------------------------- inference
@torch.no_grad()
def predict_scene(model: torch.nn.Module, scene: Scene, mean, std, chip: int = 224, overlap: int = 64,
                  batch: int = 8, device: str | None = None) -> np.ndarray:
    """Sliding-window prediction over the whole composite, averaging softmax on overlaps."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    mean_t = torch.tensor(mean, dtype=torch.float32)[:, None, None]
    std_t = torch.tensor(std, dtype=torch.float32)[:, None, None]
    _, h, w = scene.image.shape
    probs = np.zeros((N_CLASSES, h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    # Tapered window: centre pixels count more than chip edges (fewer seam artefacts).
    ramp = np.minimum(np.arange(chip) + 1, np.arange(chip)[::-1] + 1).astype(np.float32)
    taper = np.minimum.outer(np.minimum(ramp, overlap), np.minimum(ramp, overlap))
    step = chip - overlap
    rows = list(range(0, max(h - chip, 0) + 1, step)) + ([h - chip] if h > chip and (h - chip) % step else [])
    cols = list(range(0, max(w - chip, 0) + 1, step)) + ([w - chip] if w > chip and (w - chip) % step else [])
    windows = [(r, c) for r in rows for c in cols]
    for i in range(0, len(windows), batch):
        part = windows[i:i + batch]
        x = torch.stack([normalise(_pad(scene.image[:, r:r + chip, c:c + chip], chip), mean_t, std_t)
                         for r, c in part]).to(device)
        out = model(x)
        p = torch.softmax(getattr(out, "output", out), dim=1).float().cpu().numpy()
        for (r, c), pi in zip(part, p):
            hh, ww = min(chip, h - r), min(chip, w - c)
            probs[:, r:r + hh, c:c + ww] += pi[:, :hh, :ww] * taper[:hh, :ww]
            weight[r:r + hh, c:c + ww] += taper[:hh, :ww]
    pred = probs.argmax(0).astype(np.uint8)
    nodata = (scene.image == S2_NODATA).any(0) | (weight == 0)
    pred[nodata] = NODATA
    return pred


def _pad(img: np.ndarray, size: int) -> np.ndarray:
    c, h, w = img.shape
    if h == size and w == size:
        return img
    out = np.full((c, size, size), S2_NODATA, dtype=img.dtype)
    out[:, :h, :w] = img
    return out


def load_model(ckpt_path: str | Path, cfg: TrainConfig) -> torch.nn.Module:
    task = build_task(TrainConfig(**{**asdict(cfg), "pretrained": False}))
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    task.load_state_dict(state)
    return task.model
