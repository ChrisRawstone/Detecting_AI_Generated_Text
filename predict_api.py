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



from google.cloud import storage
import os

def download_gcs_folder(bucket_name, source_folder):
    """Downloads a folder from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=source_folder)  # Get list of files
    for blob in blobs:
        os.makedirs(os.path.dirname(blob.name), exist_ok=True)
        blob.download_to_filename(blob.name) 

bucket_name = 'ai-detection-bucket'
source_folder = 'models/latest' 
download_gcs_folder(bucket_name, source_folder)

model = DistilBertForSequenceClassification.from_pretrained(f"models/latest", num_labels=2)
model.to(device)


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict/") 
async def predict(text: str):
    """
    Inference endpoint          
    """
    return predict_string(model,text,device)

 




# use this command to run the post request
# curl -X 'POST' "http://127.0.0.1:8000/predict/?text=some%20random%20text"



   