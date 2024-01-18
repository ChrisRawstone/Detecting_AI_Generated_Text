import os 
import pandas as pd
from google.cloud import storage

def upload_to_gcs(local_model_dir: str, bucket_name: str, gcs_path: str, file_name: str, specific_file: str = ''):
    client = storage.Client()
    folder_name = f"{gcs_path}/{file_name}"
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(folder_name)
    if specific_file: 
        local_file_path = os.path.join(local_model_dir, specific_file)
        blob.upload_from_filename(local_file_path)
    else: 
        for local_file in os.listdir(os.path.join(local_model_dir, file_name)):
            local_file_path = os.path.join(local_model_dir, file_name, local_file)
            remote_blob_name = os.path.join(folder_name, local_file)
            # Upload the file to GCS
            blob = bucket.blob(remote_blob_name)
            blob.upload_from_filename(local_file_path)

def download_gcs_folder(source_folder: str, specific_file: str='', bucket_name: str = "ai-detection-bucket"):
    """Downloads a folder from the bucket."""
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(bucket_name)

    if specific_file:
        blob = bucket.blob(os.path.join(source_folder, specific_file))
        os.makedirs(os.path.dirname(blob.name), exist_ok=True)
        blob.download_to_filename(blob.name)
    else:
        blobs = bucket.list_blobs(prefix=source_folder)  # Get list of files
        for blob in blobs:
            os.makedirs(os.path.dirname(blob.name), exist_ok=True)
            blob.download_to_filename(blob.name)

def load_model(model_name: str = "latest", source_folder: str = "models", device = "cpu"):
    from transformers import DistilBertForSequenceClassification
    source_path = os.path.join(source_folder, model_name)

    if not os.path.exists(f"models/{model_name}"):
        download_gcs_folder(source_path)

    model = DistilBertForSequenceClassification.from_pretrained(source_path, num_labels=2)
    model.to(device)
    return model

def load_csv(file_name: str = "train.csv", source_folder: str = "data/processed/csv_files/medium_data"):
    # make directory if not exists oneline
    os.makedirs(source_folder, exist_ok=True)
    download_gcs_folder(source_folder, speficic_file=file_name)
    df = pd.read_csv(os.path.join(source_folder, file_name))
    return df