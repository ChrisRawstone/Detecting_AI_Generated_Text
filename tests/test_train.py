import os
import sys
import pytest
from mypaths import PROJECT_ROOT
sys.path.append(PROJECT_ROOT)


def test_latest_folder_existence():
    latest_folder_path = os.path.join('models', 'latest')
    assert os.path.exists(latest_folder_path), f"The 'latest' directory does not exist in the model directory. Did you train a model or pull from DVC."