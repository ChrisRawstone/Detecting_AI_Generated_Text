from mypaths import _PATH_DATA
import sys
import pytest
import os.path
sys.path.append(_PATH_DATA)
from datasets import load_dataset

# Debugging: Print the absolute file path
file_path = os.path.join(_PATH_DATA, 'data/processed/train.csv')
print("Checking for file at:", file_path)

@pytest.mark.skipif(not os.path.exists('data/processed/train.csv'), reason="Required data files not found")
def test_data_split():
    X_train = load_dataset("data/processed/train.csv")
    X_test = load_dataset("data/processed/train.csv")
    X_val = load_dataset("data/processed/validation.csv")


# test predict torch dimension

