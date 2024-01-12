
import torch
import pytest
from unittest.mock import patch
from hydra import initialize, compose
import sys
from mypaths import PROJECT_ROOT
from omegaconf import DictConfig, OmegaConf
import hydra
import os
sys.path.append(PROJECT_ROOT)
from src.train_model import main

@hydra.main(version_base=None, config_path="../src/config", config_name="default_config.yaml")
def test_model_device_assignment(config):
    print(config.experiment)

if __name__ == "__main__":
    test_model_device_assignment()