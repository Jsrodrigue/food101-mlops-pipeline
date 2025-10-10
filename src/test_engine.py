import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.eval_utils import eval_one_epoch
import torch.nn as nn
import numpy as np


def evaluate_run_and_save_results(
    run,
    model,
    dataloader,
    class_names=None,
    loss_fn_name="CrossEntropyLoss",
    metrics_list=None,
    device=None,
    save_cm_image=True,
):
    """
    Evaluate a single run, update best_model_info.json with test results,
    and optionally save confusion matrix as image in artifacts/plots.

    Args:
        run (dict): Run info containing 'run_path' and 'hyperparameters'.
        model (torch.nn.Module): Model ready to evaluate (already loaded with state_dict).
        dataloader (torch.utils.data.DataLoader): Test dataloader for this run.
        class_names (list[str], optional): List of class names in order of label indices.
        metrics_list (list[str], optional): List of metrics to compute. Defaults to ["accuracy"].
        device (torch.device, optional): Device to run evaluation on.
        save_cm_image (bool, optional): Whether to save the confusion matrix as an image. Default True.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_list = metrics_list or ["accuracy"]

    model.to(device)
    model.eval()

    # Loss function
    if loss_fn_name.lower() == "crossentropyloss":
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn_name}")

    # Evaluate with confusion matrix
    results = eval_one_epoch(
        model, dataloader, loss_fn, device, metrics_list, return_confusion_matrix=True
    )

    # ---------------- Path del archivo best_model_info.json ----------------
    run_path = Path(run["run_path"])
    info_path = run_path / "artifacts" / "best_model_info" / "best_model_info.json"

    if not info_path.exists():
        raise FileNotFoundError(f"{info_path} not found!")

    # Leer info existente
    with open(info_path, "r") as f:
        info_data = json.load(f)

    # Guardar resultados de test
    info_data["test_metrics"] = results

    # Sobrescribir el archivo
    with open(info_path, "w") as f:
        json.dump(info_data, f, indent=4)
    print(f"[INFO] Updated test results in {info_path}")

    # ---------------- Guardar matriz de confusión en plots ----------------
    if save_cm_image:
        cm = np.array(results.get("confusion_matrix"))
        if cm is not None:
            plots_dir = run_path / "artifacts" / "plots"
            plots_dir.mkdir(exist_ok=True, parents=True)
            cm_path = plots_dir / "confusion_matrix.png"

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt=".1f",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(cm_path)
            plt.close()
            print(f"[INFO] Saved confusion matrix image to {cm_path}")

    return results
