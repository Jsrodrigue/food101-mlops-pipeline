import datetime
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch import nn, optim

from src.data_setup import create_dataloader_from_folder
from src.models import EfficientNetModel, MobileNetV2Model
from src.train_engine import train_mlflow
from src.utils.model_utils import get_model_transforms


sys.path.append(str(Path(__file__).resolve().parent.parent))


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run_{timestamp}"
    cfg.outputs.run_name = run_name

    # ---------------- DATA PATHS ----------------
    train_dir = Path(cfg.dataset.train_dir)
    val_dir = Path(cfg.dataset.val_dir)

    # ---------------- TRANSFORMS ----------------

    train_transform = get_model_transforms(
        model_name=cfg.model.name.lower(),
        version=cfg.model.version,
        augmentation=cfg.train.augmentation,
    )

    val_transform = get_model_transforms(
        model_name=cfg.model.name.lower(), version=cfg.model.version, augmentation=None
    )

    # ---------------- DATALOADERS ----------------

    train_loader, class_names = create_dataloader_from_folder(
        data_dir=train_dir,
        batch_size=cfg.train.batch_size,
        transform=train_transform,
        subset_percentage=cfg.train.subset_percentage,
        seed=cfg.train.seed,
    )

    val_loader, _ = create_dataloader_from_folder(
        data_dir=val_dir,
        batch_size=cfg.train.batch_size,
        transform=val_transform,
        subset_percentage=1,
        seed=cfg.train.seed,
    )

    num_classes = len(class_names)

    # ---------------- DEVICE ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- MODEL ----------------
    if cfg.model.name.lower() == "mobilenet":
        if cfg.model.version.lower() == "v2":
            model = MobileNetV2Model(
                num_classes=num_classes, pretrained=cfg.model.pretrained
            )
        else:
            raise ValueError(
                f"[ERROR] MobileNet with version {cfg.model.version} not supported"
            )
    elif cfg.model.name.lower() == "efficientnet":
        model = EfficientNetModel(
            version=cfg.model.version,
            num_classes=num_classes,
            pretrained=cfg.model.pretrained,
        )
    else:
        raise ValueError(
            f"[ERROR] {cfg.model.name.lower()} with version {cfg.model.version} not supported"
        )

    model.freeze_backbone()

    # ---------------- OPTIMIZER & LOSS ----------------
    optimizer = optim.Adam(model.model.parameters(), lr=cfg.train.optimizer.lr)
    if cfg.train.loss_fn.lower() == "crossentropyloss":
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function: {cfg.train.loss_fn}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- TRAIN ----------------
    print("[INFO] Starting training...")
    train_mlflow(model, train_loader, val_loader, optimizer, loss_fn, cfg, device)


if __name__ == "__main__":
    main()
