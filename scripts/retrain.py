"""
Continue training a previously logged MLflow run using Hydra configs.

Usage example:
---------------
python -m scripts.continue_train retrain.run_path=mlruns/957336677953857744/5007f95823ef4f8086a506bafdf74444 retrain.epochs_extra=2
"""

from pathlib import Path
import json
import torch
from torch import nn, optim

import hydra
from hydra.utils import get_original_cwd
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.model_utils import load_model, get_model_transforms
from src.data_setup import create_dataloader_from_folder
from src.train_engine import train_mlflow


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    if not cfg.retrain.run_path:
        raise ValueError("Debe especificar retrain.run_path")

    run_path = Path(get_original_cwd()) / cfg.retrain.run_path
    best_info_path = run_path / "artifacts/best_model_info/best_model_info.json"
    state_dict_path = run_path / "artifacts/model/model_state_dict.pth"

    if not best_info_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo {best_info_path}")

    with open(best_info_path, "r") as f:
        best_info = json.load(f)

    # Solo sobrescribimos los hyperparameters que ya están definidos en cfg.train
    for k, v in best_info["hyperparameters"].items():
        if k in cfg.train:
            cfg.train[k] = v

    # --- Sobrescribir dinámicamente los valores si están en null ---
    cfg.retrain.batch_size = cfg.retrain.batch_size or best_info["hyperparameters"].get(
        "batch_size", 64
    )
    cfg.dataset.train_dir = cfg.retrain.train_dir or best_info.get(
        "train_dir", "data/dataset/train"
    )
    cfg.dataset.val_dir = cfg.retrain.val_dir or best_info.get(
        "val_dir", "data/dataset/val"
    )
    cfg.train.epochs = cfg.retrain.epochs_extra or 2

    # ---------------- DATA PATHS ----------------
    train_dir = Path(cfg.dataset.train_dir)
    val_dir = Path(cfg.dataset.val_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Loaders ---
    train_transform = get_model_transforms(
        model_name=best_info["hyperparameters"]["model_name"].lower(),
        version=best_info["hyperparameters"].get("version"),
        augmentation=best_info["hyperparameters"].get("augmentation"),
    )
    val_transform = get_model_transforms(
        model_name=best_info["hyperparameters"]["model_name"].lower(),
        version=best_info["hyperparameters"].get("version"),
        augmentation=None,
    )

    train_loader, class_names = create_dataloader_from_folder(
        data_dir=Path(get_original_cwd()) / train_dir,
        batch_size=cfg.retrain.batch_size,
        transform=train_transform,
        subset_percentage=best_info["hyperparameters"].get("subset_percentage", 1.0),
        seed=42,
    )
    val_loader, _ = create_dataloader_from_folder(
        data_dir=Path(get_original_cwd()) / val_dir,
        batch_size=cfg.retrain.batch_size,
        transform=val_transform,
        subset_percentage=1.0,
        seed=42,
    )
    num_classes = len(class_names)

    # --- Load Model ---
    model = load_model(
        state_dict_path=state_dict_path,
        model_name=best_info["hyperparameters"]["model_name"],
        num_classes=num_classes,
        version=best_info["hyperparameters"].get("version"),
        device=device,
    )

    if model is None:
        raise RuntimeError("No se pudo cargar el modelo")

    # --- Optimizer & Loss ---
    optimizer = optim.Adam(
        model.parameters(), lr=best_info["hyperparameters"].get("optimizer_lr", 1e-3)
    )
    loss_fn = nn.CrossEntropyLoss()

    # --- Training ---
    print(f"[INFO] Continuing training from run {run_path}")
    train_mlflow(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        cfg=cfg,
        device=device,
        continue_training=True,
        continue_path=run_path,
    )


if __name__ == "__main__":
    main()
