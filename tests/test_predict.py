import sys
import pytest
import os.path
import pandas as pd
from mypaths import PROJECT_ROOT

sys.path.append(PROJECT_ROOT)
from src.predict_model import find_latest_folder

model_name = find_latest_folder("models")


@pytest.mark.skipif(
    not os.path.exists(f"results/predictions_{model_name}.json"),
    reason="Prediction file not found. Run 'python src/predict_model.py' to generate predictions.",
)
def test_predict_model():
    # Read the predictions from json file as table
    predictions_dataframe = pd.read_json(f"results/predictions_{model_name}.json", orient="table")

    assert "text" in predictions_dataframe.columns
    assert "prediction" in predictions_dataframe.columns
    assert "generated" in predictions_dataframe.columns

    assert len(predictions_dataframe) >= 1
    assert set(predictions_dataframe["prediction"].unique()) == {0, 1}
    assert set(predictions_dataframe["generated"].unique()) == {0, 1}


if __name__ == "__main__":
    test_predict_model()
