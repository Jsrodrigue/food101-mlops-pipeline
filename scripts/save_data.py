import hydra
from omegaconf import DictConfig

from src.data_engine import create_data


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    create_data(cfg)


if __name__ == "__main__":
    main()
