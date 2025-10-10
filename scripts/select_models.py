import hydra
from omegaconf import DictConfig
from pathlib import Path

from src.utils.runs_utils import select_top_k_runs


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Select the top-K model runs based on a specified metric,
    copy their artifacts to a target directory, and
    add class names from class_map.json to each best_model_info.json.
    """
    project_root = Path(hydra.utils.get_original_cwd())

    select_top_k_runs(
        runs_root=project_root / cfg.select_model.source_runs_dir,
        target_dir=project_root / cfg.select_model.target_selected_models_dir,
        top_k=cfg.select_model.top_k,
        metric_name=cfg.select_model.metric_name,
        metric_source=getattr(cfg.select_model, "metric_source", "val_metrics"),
        higher_better=False if cfg.select_model.metric_name == "loss" else True,
        class_map_path=project_root / "data" / "class_map.json",
    )


if __name__ == "__main__":
    main()
