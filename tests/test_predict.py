import os.path

import pytest
from datasets import load_from_disk

from src.predict_model import predict
from src.utils import load_model


@pytest.mark.skipif(
    not os.path.exists(f"data/processed/tokenized_data/small_data/test_dataset_tokenized"),
    reason="Prediction file not found. Run 'python src/predict_model.py' to generate predictions.",
)
def test_predict_model():
    # Read the predictions from json file as table
    model = load_model()
    tokenized_dataset = load_from_disk("data/processed/tokenized_data/small_data/test_dataset_tokenized")

    predictions_dataframe = predict(model, tokenized_dataset, "cpu")

    assert "text" in predictions_dataframe.columns, "text not found in predictions_dataframe.columns"
    assert "prediction" in predictions_dataframe.columns, "prediction not found in predictions_dataframe.columns"
    assert "label" in predictions_dataframe.columns, "label not found in predictions_dataframe.columns"

    assert len(predictions_dataframe) >= 1, "No predictions found in predictions_dataframe"
    assert set(predictions_dataframe["prediction"].unique()) <= {0, 1}, "Invalid values in predictions_dataframe"
    assert set(predictions_dataframe["label"].unique()) <= {0, 1}, "Invalid values in predictions_dataframe"


if __name__ == "__main__":
    test_predict_model()
