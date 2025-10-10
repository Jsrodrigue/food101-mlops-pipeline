from pathlib import Path

import hydra
from omegaconf import DictConfig
from torch import cuda
from tqdm import tqdm  # <--- barra de progreso

from src.test_engine import evaluate_run_and_save_results
from src.utils.data_setup_utils import create_test_loaders
from src.utils.model_utils import load_model
from src.utils.runs_utils import collect_model_transforms, find_runs_in_dir


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):

    # ---------------- DATA PATHS ----------------
    test_dir = Path(cfg.dataset.test_dir)

    # -------------GET RUNS INFO -----------------
    runs = find_runs_in_dir(Path.cwd() / cfg.test.runs_dir)
    print(f"[INFO] Testing {len(runs)} models...")

    # --------------TRANSFORMS-----------------------
    transforms_dicts = collect_model_transforms(runs)

    # ------------ TEST LOADERS ----------------------
    loaders_dict, class_names = create_test_loaders(
        transforms_dicts, test_dir, batch_size=cfg.test.batch_size
    )

    # ----------- LOSS_FN, METRICS and DEVICE----------------
    loss_fn_name = cfg.test.loss_fn
    metrics_list = cfg.test.metrics
    device = "cuda" if cuda.is_available() else "cpu"

    print(f"[INFO] Using loss function: {loss_fn_name}")
    print(f"[INFO] Using metrics: {metrics_list}")
    print(f"[INFO] Device: {device}")

    # ----------- EVALUATION LOOP with progress bar --------------------------
    for run in tqdm(runs, desc="Evaluating models", unit="model"):
        hp = run["hyperparameters"]
        key = (hp["model_name"], hp.get("version"))
        dataloader = loaders_dict[key]

        print(
            f"\n[INFO] Evaluating run: {run['run_path'].name} "
            f"(Model: {hp['model_name']}, Version: {hp.get('version')})"
        )

        # Load model
        model = load_model(
            state_dict_path=run["state_dict"],
            model_name=hp["model_name"],
            version=hp.get("version", None),
            num_classes=len(class_names),
            device=device,
        )

        # Evaluate and save results
        results = evaluate_run_and_save_results(
            run,
            model,
            dataloader,
            class_names=class_names,
            loss_fn_name=loss_fn_name,
            metrics_list=metrics_list,
            device=device,
            save_cm_image=cfg.test.save_cm_img,
        )

        # Show summary in console
        print("[RESULTS] Metrics:")
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: (array or list, not displayed)")

        # Print links to saved files
        test_dir = Path(run["run_path"]) / "test"
        json_path = test_dir / "test.json"
        cm_path = test_dir / "confusion_matrix.png"

        print(f"[INFO] JSON results: file://{json_path.resolve()}")
        if cm_path.exists():
            print(f"[INFO] Confusion Matrix image: file://{cm_path.resolve()}")
        else:
            print("[INFO] Confusion Matrix image not saved.")

        # Free GPU memory after evaluation
        model.to("cpu")
        cuda.empty_cache()


if __name__ == "__main__":
    main()
