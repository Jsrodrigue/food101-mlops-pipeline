import hashlib
import json
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from .model_utils import get_model_transforms


def collect_model_transforms(runs: List[Dict]) -> Dict[Tuple[str, str], Callable]:
    """
    Collect default torchvision transforms for a list of trained model runs.

    Each run is expected to have a 'hyperparameters' dict with keys 'model_name' and optionally 'version'.

    Args:
        runs (list): List of run dictionaries, each containing 'hyperparameters'.

    Returns:
        Dict[Tuple[str, str], Callable]: Maps (model_name, version) to torchvision transforms.
    """
    transforms_dict = {}

    for run in runs:
        hp = run.get("hyperparameters", {})
        model_name = hp.get("model_name")
        version = hp.get("version", None)

        if not model_name:
            print(f"[WARN] Run {run.get('run_path')} missing 'model_name', skipping...")
            continue

        try:
            transforms_dict[(model_name, version.lower() if version else None)] = (
                get_model_transforms(
                    model_name=model_name, version=version, augmentation=False
                )
            )
            print(
                f"[INFO] Collected transforms for model: {model_name}, version: {version}"
            )
        except Exception as e:
            print(
                f"[WARN] Could not get transforms for model '{model_name}' "
                f"{'version '+version if version else ''}: {e}"
            )

    if not transforms_dict:
        print("[WARN] No transforms were collected from the provided runs.")

    return transforms_dict


def find_runs_in_dir(models_root: Path) -> List[Dict]:
    """
    Discover all model runs inside a directory.
    Each run is expected to have:
        - artifacts/best_model_info/best_model_info.json
        - artifacts/model/model_state_dict.pth

    Args:
        models_root (Path | str): Root directory containing runs.

    Returns:
        List[Dict]: Each run dictionary contains:
            - "run_path": Path to run folder
            - "hyperparameters": Dict from best_model_info.json
            - "info": Full contents of best_model_info.json
            - "state_dict": Path to model_state_dict.pth
    """
    models_root = Path(models_root)
    runs = []

    for info_file in models_root.rglob(
        "artifacts/best_model_info/best_model_info.json"
    ):
        state_dict_file = info_file.parent.parent / "model" / "model_state_dict.pth"
        if not state_dict_file.exists():
            continue
        with open(info_file, "r") as f:
            info = json.load(f)
        run_dir = info_file.parent.parent.parent
        runs.append(
            {
                "run_path": run_dir,
                "hyperparameters": info.get("hyperparameters", {}),
                "info": info,
                "state_dict": state_dict_file,
            }
        )
    return runs


def select_top_k_runs(
    runs_root: str,
    target_dir: str,
    top_k: int = 5,
    metric_name: str = "loss",
    metric_source: str = "val_metrics",  # NEW: "train_metrics" o "val_metrics"
    higher_better: bool = False,
    copy_plots: bool = True,
    class_map_path: str = "data/dataset/class_map.json",
):
    """
    Select top-K model runs based on a metric from train_metrics or val_metrics,
    copy their artifacts to a target directory, and add class_names to
    best_model_info.json without removing existing info.
    """
    runs_root = Path(runs_root)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Load class names
    class_names = []
    class_map_file = Path(class_map_path)
    if class_map_file.exists():
        with open(class_map_file, "r") as f:
            class_map = json.load(f)
        class_names = [
            class_map["idx_to_class"][str(i)]
            for i in range(len(class_map["idx_to_class"]))
        ]
        print(f"[INFO] Loaded {len(class_names)} class names")
    else:
        print(f"[WARN] Class map not found: {class_map_file}")

    # Find all runs
    runs = find_runs_in_dir(runs_root)

    # Filter runs with the specified metric from the given source
    filtered = []
    for run in runs:
        metric_val = run.get("info", {}).get(metric_source, {}).get(metric_name)
        if metric_val is not None:
            run["metric_value"] = metric_val
            filtered.append(run)

    if not filtered:
        print(f"[WARN] No runs with metric '{metric_name}' found in '{metric_source}'")
        return

    # Sort and select top-K
    sorted_runs = sorted(
        filtered, key=lambda x: x["metric_value"], reverse=higher_better
    )
    top_runs = sorted_runs[:top_k]

    # Copy artifacts and update best_model_info.json
    for run in top_runs:
        hp = run.get("hyperparameters", {})
        model_name = hp.get("model_name", "model")
        version = hp.get("version", "v0")
        run_hash = hashlib.sha1(str(run["run_path"]).encode()).hexdigest()[:5]
        dest = target_dir / f"{model_name}_{version}_{run_hash}"
        dest.mkdir(parents=True, exist_ok=True)

        artifacts_src = run["run_path"] / "artifacts"
        for item in artifacts_src.iterdir():
            if item.name == "plots" and not copy_plots:
                continue
            dest_item = dest / "artifacts" / item.name
            if item.is_dir():
                shutil.copytree(item, dest_item, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest_item)

        # Update best_model_info.json without removing metrics
        info_file = dest / "artifacts" / "best_model_info" / "best_model_info.json"
        if info_file.exists():
            with open(info_file, "r") as f:
                info = json.load(f)

            if class_names:
                info["class_names"] = class_names

            with open(info_file, "w") as f:
                json.dump(info, f, indent=4)

            print(f"[INFO] Updated best_model_info.json at {info_file}")
