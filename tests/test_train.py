
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

@pytest.fixture
def models_folder():
    return 'models'

def test_latest_folder_existence(models_folder):
    latest_folder_path = os.path.join(models_folder, 'latest')
    assert os.path.exists(latest_folder_path), f"The 'latest' folder does not exist in '{models_folder}'."