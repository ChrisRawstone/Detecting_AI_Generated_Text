import os
import sys
import pytest
from mypaths import PROJECT_ROOT
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf 
from src.train_model import train


# hydra_logger = hydra.utils.log  # Use Hydra logger for logging

# # Load metric for evaluation
# metric = load_metric("accuracy")


print("YOOYOOYOOYOY ",PROJECT_ROOT)
sys.path.append(PROJECT_ROOT+"/src")



def test_latest_folder_existence():
    print(os.getcwd())

    # @hydra.main(config_path="config", config_name="default_config.yaml")
    with initialize(version_base=None, config_path="../src/config"):
        cfg = compose(config_name="default_config.yaml")
        
    train(cfg,push_model_to_gcs=False)


    latest_folder_path = os.path.join("models", "latest")
    assert os.path.exists(
        latest_folder_path
    ), f"The 'latest' directory does not exist in the model directory. Did you train a model?"


if __name__ == "__main__":
    test_latest_folder_existence()