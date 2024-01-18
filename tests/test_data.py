import sys
import pytest
import os.path
import pandas as pd
from mypaths import PROJECT_ROOT

sys.path.append(PROJECT_ROOT)
from src.data.make_dataset import make_dataset, get_data


def test_get_data():
    with pytest.raises(FileNotFoundError, match="No generated data found. Please run DVC pull."):
        get_data(10, "Not/existing/path")


@pytest.mark.skipif(
    not os.path.exists("data/processed/csv_files/small_data/train.csv"), reason="Required data files not found"
)
def test_make_dataset():
    X_train = pd.read_csv("data/processed/csv_files/small_data/train.csv")
    X_test = pd.read_csv("data/processed/csv_files/small_data/test.csv")
    X_val = pd.read_csv("data/processed/csv_files/small_data/validation.csv")

    # Assert dataset sizes
    assert len(X_train) >= 1, "Training dataset did not have at least one sample"
    assert len(X_test) >= 1, "Test dataset did not have at least one sample"
    assert len(X_val) >= 1, "Validation dataset did not have at least one sample"


if __name__ == "__main__":
    test_get_data()
    test_make_dataset()
