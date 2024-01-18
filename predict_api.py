import os
import pandas as pd
import torch
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from transformers import DistilBertForSequenceClassification
from google.cloud import storage
from src.predict_model import predict_string, predict_csv
# from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
bucket_name = "ai-detection-bucket"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def download_gcs_folder(source_folder: str):
    """Downloads a folder from the bucket."""
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=source_folder)  # Get list of files
    for blob in blobs:
        os.makedirs(os.path.dirname(blob.name), exist_ok=True)
        blob.download_to_filename(blob.name)


def load_model(model_name: str = "latest", source_folder: str = "models"):
    source_path = os.path.join(source_folder, model_name)
    download_gcs_folder(source_path)
    model = DistilBertForSequenceClassification.from_pretrained(source_path, num_labels=2)
    model.to(device)
    return model

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict_string/") 
async def predict_string(text: str, model_name: str = "latest"):
    """
    Inference endpoint          
    """
    model = load_model(model_name = model_name)

    return predict_string()


@app.post("/process_csv/")
async def process_csv(file: UploadFile = File(...), model_name: str = "latest", true_label_provided: bool = False):
    temp_file_path = "tempfile.csv"
    with open(temp_file_path, 'wb') as buffer:
        content = await file.read()  # Read the file content
        buffer.write(content)  # Write to a temporary file

    # Read the CSV into a DataFrame
    df = pd.read_csv(temp_file_path)  

    model = load_model(model_name = model_name)

    # Make predictions 
    predictions_df = predict_csv(model, df, device)

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_predictions_file = f"results/predictions_{timestamp}.csv"
    predictions_df.to_csv(local_predictions_file, index=False)

    # Upload to GCS
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(bucket_name)
    gcs_file_path = f"inference_predictions/predictions_{timestamp}.csv"
    blob = bucket.blob(gcs_file_path)
    blob.upload_from_filename(local_predictions_file)

    # do different return depending on if true label provided or not
    if true_label_provided:
        # do 



    return FileResponse("results/predictions.csv", media_type="text/csv", filename="predictions.csv")

# Instrumentator().instrument(app).expose(app)



 




# use this command to run the post request
# curl -X 'POST' "http://127.0.0.1:8000/predict/?text=some%20random%20text"