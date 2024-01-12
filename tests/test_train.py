import os
import sys
import pytest
from mypaths import PROJECT_ROOT
sys.path.append(PROJECT_ROOT)

@pytest.fixture
def models_folder():
    return 'models'

def test_latest_folder_existence(models_folder):
    latest_folder_path = os.path.join(models_folder, 'latest')
    assert os.path.exists(latest_folder_path), f"The 'latest' folder does not exist in '{models_folder}'."