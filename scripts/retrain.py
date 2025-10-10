# retrain.py
import datetime
from pathlib import Path
import sys
import torch
from torch import nn, optim
import hydra
from omegaconf import DictConfig

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_setup import create_dataloader_from_folder
from src.models import EfficientNetModel, MobileNetV2Model
from src.train_engine import train_mlflow
from src.utils.model_utils import get_model_transforms


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    # ------------------ RUN INFO ------------------
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"retrain_{timestamp}"
    cfg.outputs.run_name = run_name

    retrain_dir = Path.cwd() / "retrained_models" / run_name
    retrain_dir.mkdir(parents=True, exist_ok=True)

    # ------------------ DATALOADERS -------------
    train_loader, class_names = create_dataloader_from_folder(
        data_dir=Path(cfg.dataset.train_dir),
        batch_size=cfg.train.batch_size,
        transform=get_model_transforms(
            model_name=cfg.model.name.lower(),
            version=cfg.model.version,
            augmentation=cfg.train.augmentation,
        ),
        subset_percentage=cfg.train.subset_percentage,
        seed=cfg.train.seed,
    )

    val_loader, _ = create_dataloader_from_folder(
        data_dir=Path(cfg.dataset.val_dir),
        batch_size=cfg.train.batch_size,
        transform=get_model_transforms(
            model_name=cfg.model.name.lower(),
            version=cfg.model.version,
            augmentation=None,
        ),
        subset_percentage=1.0,
        seed=cfg.train.seed,
    )

    num_classes = len(class_names)

    # ------------------ MODEL ------------------
    if cfg.model.name.lower() == "mobilenet":
        model = MobileNetV2Model(
            num_classes=num_classes, pretrained=cfg.model.pretrained
        )
    else:
        model = EfficientNetModel(
            version=cfg.model.version,
            num_classes=num_classes,
            pretrained=cfg.model.pretrained,
        )

    model.freeze_backbone()
    if cfg.train.unfreeze_layers > 0:
        model.unfreeze_backbone(cfg.train.unfreeze_layers)

    # ------------------ OPTIMIZER & LOSS ------------------
    optimizer = optim.Adam(model.model.parameters(), lr=cfg.train.optimizer.lr)
    if cfg.train.loss_fn.lower() == "crossentropyloss":
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function: {cfg.train.loss_fn}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------ TRAIN ------------------
    print(f"[INFO] Starting retraining... Results will be saved in {retrain_dir}")
    train_mlflow(model.model, train_loader, val_loader, optimizer, loss_fn, cfg, device)

    # ------------------ SAVE MODEL ------------------
    model_path = retrain_dir / "model_state_dict.pth"
    torch.save(model.model.state_dict(), model_path)
    print(f"[INFO] Retrained model saved at {model_path}")


if __name__ == "__main__":
    main()
