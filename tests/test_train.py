import pytest
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
from mypaths import PROJECT_ROOT
sys.path.append(PROJECT_ROOT)
from src.train_model import main


@hydra.main(version_base=None, config_path="src/config", config_name="default_config.yaml")
def test_model_device_assignment(config):
    main(config)



@pytest.fixture
def models_folder():
    return 'models'

def test_latest_folder_existence(models_folder):
    latest_folder_path = os.path.join(models_folder, 'latest')
    assert os.path.exists(latest_folder_path), f"The 'latest' folder does not exist in '{models_folder}'."
