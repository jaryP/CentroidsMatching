import hydra
from omegaconf import DictConfig

from training import avalanche_training


@hydra.main(config_path="configs",
            config_name="config")
def my_app(cfg: DictConfig) -> None:
    avalanche_training(cfg)


if __name__ == "__main__":
    my_app()
