import os
import pandas as pd
import torch
from fastapi import BackgroundTasks, FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from google.cloud import storage
from src.predict_model import predict_string, predict_csv
from http import HTTPStatus

app = FastAPI()

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def download_gcs_folder(bucket_name, source_folder):
    """Downloads a folder from the bucket."""
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=source_folder)  # Get list of files
    for blob in blobs:
        os.makedirs(os.path.dirname(blob.name), exist_ok=True)
        blob.download_to_filename(blob.name)


bucket_name = "ai-detection-bucket"
source_folder = "models/latest"
#download_gcs_folder(bucket_name, source_folder)

model = DistilBertForSequenceClassification.from_pretrained(f"models/latest", num_labels=2)
model.to(device)


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict_string/") 
async def predict_string(text: str):
    """
    Inference endpoint          
    """
    return predict_string(model, text, device)


@app.post("/process_csv/")
async def process_csv(file: UploadFile = File(...)):
    temp_file_path = "tempfile.csv"
    with open(temp_file_path, 'wb') as buffer:
        content = await file.read()  # Read the file content
        buffer.write(content)  # Write to a temporary file

    # Read the CSV into a DataFrame
    df = pd.read_csv(temp_file_path)  

    # Make predictions 
    predictions_df = predict_csv(model, df, device)
    predictions_df.to_csv("results/predictions.csv", index=False)

    return FileResponse("results/predictions.csv", media_type="text/csv", filename="predictions.csv")





 




# use this command to run the post request
# curl -X 'POST' "http://127.0.0.1:8000/predict/?text=some%20random%20text"