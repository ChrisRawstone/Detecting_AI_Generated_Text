import sys
import pytest
import os.path
import pandas as pd
from mypaths import PROJECT_ROOT
sys.path.append(PROJECT_ROOT)
from src.data.make_dataset import make_dataset, get_data

def test_get_data():
    with pytest.raises(FileNotFoundError, match='No generated data found. Please run dvc pull.'):
        get_data(10,'Not/existing/path')


@pytest.mark.skipif(not os.path.exists('data/processed/train.csv'), reason="Required data files not found")
def test_make_dataset():

    X_train = pd.read_csv("data/processed/train.csv")
    X_test = pd.read_csv("data/processed/test.csv")
    X_val = pd.read_csv("data/processed/validation.csv")

    N_train = 20
    N_test = 20
    N_val = 20

    # Assert dataset sizes
    assert len(X_train) >= N_train, "Training dataset did not have the expected number of samples"
    assert len(X_test) >= N_test, "Test dataset did not have the expected number of samples"
    assert len(X_val) >= N_val, "Validation dataset did not have the expected number of samples"

    # Minimum count of each label in each split
    min_label_count = 5

    # Check for label imbalance in each dataset
    for dataset, name in [(X_train, 'Train'), (X_test, 'Test'), (X_val, 'Validation')]:
        for label in [0, 1]:
            count = dataset[dataset['generated'] == label].shape[0]
            assert count >= min_label_count, f"{name} set has fewer than {min_label_count} instances of label {label}"

if __name__ == '__main__':
    test_get_data()
    test_make_dataset()