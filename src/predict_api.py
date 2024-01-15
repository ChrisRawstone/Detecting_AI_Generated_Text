from src.predict_model import predict_string
import torch
from fastapi import BackgroundTasks, FastAPI
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


app = FastAPI()

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# model = DistilBertForSequenceClassification.from_pretrained(f"models/latest", num_labels=2)
# model.to(device)

from google.cloud import storage
import os

def download_gcs_folder(bucket_name, source_folder, destination_dir):
    """Downloads a folder from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=source_folder)  # Get list of files
    for blob in blobs:
        destination_file_name = os.path.join(destination_dir, blob.name)
        os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
        blob.download_to_filename(destination_file_name)

# Example usage
bucket_name = 'ai-detection-bucket'
source_folder = 'models/latest'  # Make sure to include the trailing slash
destination_dir = ''  # Make sure to include the trailing slash

download_gcs_folder(bucket_name, source_folder, destination_dir)



# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

# @app.post("/predict/") 
# async def predict(text: str):
#     """Inference endpoint          
#     """
#     result = predict_string(model,text,device)
#     return result

 




# use this command to run the post request
# curl -X 'POST' "http://127.0.0.1:8000/predict/?text=some%20random%20text"



   