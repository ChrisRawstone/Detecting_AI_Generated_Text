import os
import sys

import hydra
import pytest
from hydra import compose, initialize
from mypaths import PROJECT_ROOT
from omegaconf import OmegaConf

from src.train_model import train

hydra_logger = hydra.utils.log


def test_latest_folder_existence():
    experiment_name = "unit_testing_hparams"
    with initialize(version_base=None, config_path="../src/config"):
        cfg = compose(config_name="default_config.yaml", overrides=[f"experiment={experiment_name}"])

    train(cfg)

    latest_folder_path = os.path.join("models", "latest")
    assert os.path.exists(
        latest_folder_path
    ), f"The 'latest' directory does not exist in the model directory. Did you train a model?"


if __name__ == "__main__":
    test_latest_folder_existence()
