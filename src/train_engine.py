# src/train_engine.py

import json
from pathlib import Path

import mlflow
import torch
from torch import optim
from tqdm import tqdm

from .utils.eval_utils import eval_one_epoch
from .utils.metrics import compute_metrics
from .utils.mlflow_utils import (
    log_hyperparams_mlflow,
    log_loss_curve_mlflow,
    setup_mlflow,
    update_best_model,
)
from .utils.plot_utils import log_loss_curve
from .utils.seed_utils import set_seed


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=5, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement for {self.counter} epoch(s)")
            if self.counter >= self.patience:
                self.early_stop = True


# -------------------- TRAIN STEP -------------------- #


def train_step(model, dataloader, loss_fn, optimizer, device, metrics_list=None):
    """
    Perform one training epoch.
    """
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y = y.long()

        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(outputs)
        all_labels.append(y)

    avg_loss = total_loss / len(dataloader)
    preds = torch.cat(all_preds).argmax(dim=1)
    labels = torch.cat(all_labels)
    metrics_dict = compute_metrics(labels, preds, metrics_list or ["accuracy"])
    metrics_dict["loss"] = avg_loss
    return metrics_dict


# -------------------- MAIN TRAIN MLflow -------------------- #
def train_mlflow(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    cfg,
    device=None,
    continue_training=False,
    continue_path=None,
    prev_metrics=None,
):
    """
    Train a PyTorch model and log metrics, plots, and best model to MLflow.

    Args:
        model: PyTorch model.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        optimizer: PyTorch optimizer.
        loss_fn: Loss function.
        cfg: Configuration dictionary or OmegaConf object.
        device: torch.device, optional.
        continue_training: bool, whether to resume from previous run.
        continue_path: Path to previous run, required if continue_training=True.
        prev_metrics: dict of previous metrics to continue logging.

    Returns:
        results: dict with training/validation metrics per epoch.
    """

    set_seed(cfg.train.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Scheduler setup ---
    scheduler = None
    if cfg.train.scheduler.type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=cfg.train.scheduler.get("patience", 3)
        )
    elif cfg.train.scheduler.type == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.train.scheduler.get("step_size", 5),
            gamma=cfg.train.scheduler.get("gamma", 0.1),
        )

    # --- MLflow setup ---
    mlflow_dir, run_name = setup_mlflow(cfg)
    early_stopper = EarlyStopping(patience=cfg.train.early_stop_patience, verbose=True)

    # --- Metrics storage ---
    if continue_training:
        results = prev_metrics or {}
        if not results:
            json_path = (
                continue_path / "artifacts" / "metrics" / "training_results.json"
            )
            metrics_dir = continue_path / "metrics"

            if json_path.exists():
                print(f"[INFO] Loading previous training results from {json_path}")
                with open(json_path, "r") as f:
                    results = json.load(f)
            else:
                print(
                    "[WARN] No training_results.json found, reconstructing from metrics folder..."
                )
                results = {f"train_{m}": [] for m in cfg.train.metrics + ["loss"]}
                results.update({f"val_{m}": [] for m in cfg.train.metrics + ["loss"]})

                if metrics_dir.exists():
                    for metric_file in metrics_dir.iterdir():
                        metric_name = metric_file.name
                        with open(metric_file, "r") as f:
                            try:
                                results[metric_name] = [
                                    float(line.strip().split()[1])
                                    for line in f
                                    if len(line.strip().split()) > 1
                                ]
                            except Exception as e:
                                print(f"[WARN] Could not parse {metric_name}: {e}")
                num_epochs = len(next(iter(results.values()), []))
                results["epochs"] = list(range(num_epochs))
                results["best_epoch"] = num_epochs - 1 if num_epochs > 0 else None
    else:
        results = {f"train_{m}": [] for m in cfg.train.metrics + ["loss"]}
        results.update({f"val_{m}": [] for m in cfg.train.metrics + ["loss"]})

    best_model_epoch = results.get("best_epoch", -1)
    best_model_loss = (
        results["val_loss"][best_model_epoch] if best_model_epoch >= 0 else float("inf")
    )

    # --- Main training loop ---
    with mlflow.start_run(run_name=run_name):
        log_hyperparams_mlflow(cfg, loss_fn)

        start_epoch = len(results["epochs"]) if continue_training else 0
        for epoch in tqdm(
            range(start_epoch, start_epoch + cfg.train.epochs), desc="Training"
        ):
            if epoch == 2 and cfg.train.unfreeze_layers > 0:
                model.unfreeze_backbone(cfg.train.unfreeze_layers)

            train_metrics = train_step(
                model, train_loader, loss_fn, optimizer, device, cfg.train.metrics
            )
            val_metrics = eval_one_epoch(
                model, val_loader, loss_fn, device, cfg.train.metrics
            )

            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics["loss"])
                else:
                    scheduler.step()

            for key in train_metrics:
                mlflow.log_metric(f"train_{key}", train_metrics[key], step=epoch)
                mlflow.log_metric(f"val_{key}", val_metrics[key], step=epoch)
                results[f"train_{key}"].append(train_metrics[key])
                results[f"val_{key}"].append(val_metrics[key])

            train_str = ", ".join([f"{k} {v:.4f}" for k, v in train_metrics.items()])
            val_str = ", ".join([f"{k} {v:.4f}" for k, v in val_metrics.items()])
            print(f"Epoch {epoch+1}: Train: {train_str}")
            print(f"        Val: {val_str}\n")

            best_model_loss, is_new_best = update_best_model(
                model,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_loss=best_model_loss,
                cfg=cfg,
                epoch=epoch,
            )
            if is_new_best:
                best_model_epoch = epoch

            early_stopper(val_metrics["loss"])
            if early_stopper.early_stop:
                print(f"[INFO] Early stopping at epoch {epoch+1}")
                break

        results["epochs"] = list(range(len(results["train_loss"])))
        results["best_epoch"] = best_model_epoch

        json_path = Path("training_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        mlflow.log_artifact(str(json_path), artifact_path="metrics")
        json_path.unlink()

        plot_path = log_loss_curve(results)
        log_loss_curve_mlflow(plot_path, artifact_path="plots")

    return results
