import os 
from google.cloud import storage

def upload_to_gcs(local_model_dir, bucket_name, gcs_path, file_name):
    client = storage.Client()
    folder_name = f"{gcs_path}/{file_name}"
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(folder_name)

    # Upload each file from local_model_dir to the GCS folder
    for local_file in os.listdir(os.path.join(local_model_dir, file_name)):
        local_file_path = os.path.join(local_model_dir, file_name, local_file)
        remote_blob_name = os.path.join(folder_name, local_file)
        # Upload the file to GCS
        blob = bucket.blob(remote_blob_name)
        blob.upload_from_filename(local_file_path)