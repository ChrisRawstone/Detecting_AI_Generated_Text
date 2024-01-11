from mypaths import _PATH_DATA
import sys
import pytest
import os.path
sys.path.append(_PATH_DATA)
from make_dataset import make_dataset

# Debugging: Print the absolute file path
file_path = os.path.join(_PATH_DATA, 'data/processed/train.csv')
print("Checking for file at:", file_path)

@pytest.mark.skipif(not os.path.exists('data/processed/train.csv'), reason="Required data files not found")

def test_data_loading():
    train_dataset, test_dataset, val_dataset = make_dataset()

    N_train = 2175
    N_test = 272
    N_val = 272 

    # Assert dataset sizes
    assert len(train_dataset) == N_train, "Training dataset did not have the expected number of samples"
    assert len(test_dataset) == N_test, "Test dataset did not have the expected number of samples"
    assert len(val_dataset) == N_val, "Validation dataset did not have the expected number of samples"
