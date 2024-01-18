import sys
import pytest
import os.path
import pandas as pd
import torch
from transformers import DistilBertForSequenceClassification
from datasets import load_from_disk
from mypaths import PROJECT_ROOT
from src.predict_model import predict

def download_model_from_gcs(local_download_dir, bucket_name, gcs_path, model_name):
    from google.cloud import storage
    import os
    client = storage.Client()
    folder_name = f"{gcs_path}/{model_name}"
    bucket = client.bucket(bucket_name)

    # Create local download directory if it doesn't exist
    os.makedirs(os.path.join(local_download_dir, model_name), exist_ok=True)

    # Download each file from the GCS folder to the local download directory
    blobs = bucket.list_blobs(prefix=folder_name)
    for blob in blobs:
        local_file_path = os.path.join(local_download_dir, model_name, os.path.basename(blob.name))
        blob.download_to_filename(local_file_path)


@pytest.mark.skipif(
    not os.path.exists(f"data/processed/tokenized_data/small_data/test_dataset_tokenized"),
    reason="Prediction file not found. Run 'python src/predict_model.py' to generate predictions.")
def test_predict_model():
    # Read the predictions from json file as table
    download_model_from_gcs("models/", "ai-detection-bucket", "models", "latest")
    device = torch.device("cpu")

    model = DistilBertForSequenceClassification.from_pretrained(f"models/latest", num_labels=2)
    model.to(device)
    tokenized_dataset = load_from_disk("data/processed/tokenized_data/small_data/test_dataset_tokenized")

    predictions_dataframe = predict(model, tokenized_dataset, device)

    assert "text" in predictions_dataframe.columns, 'text not found in predictions_dataframe.columns'
    assert "prediction" in predictions_dataframe.columns, 'prediction not found in predictions_dataframe.columns'
    assert "label" in predictions_dataframe.columns, 'label not found in predictions_dataframe.columns'

    assert len(predictions_dataframe) >= 1, "No predictions found in predictions_dataframe"
    assert set(predictions_dataframe["prediction"].unique()) <= {0, 1}, "Invalid values in predictions_dataframe"
    assert set(predictions_dataframe["label"].unique()) <= {0, 1}, "Invalid values in predictions_dataframe"

if __name__ == "__main__":
    test_predict_model()
