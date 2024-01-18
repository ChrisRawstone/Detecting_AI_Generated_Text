import os
import pandas as pd
import torch
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse
from transformers import DistilBertForSequenceClassification
from google.cloud import storage
from src.predict_model import predict_string, predict_csv
# from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


app = FastAPI()
bucket_name = "ai-detection-bucket"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


app.mount("/static", StaticFiles(directory="static"), name="static")

class TextModel(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    with open('static/index.html', 'r') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

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

    if not os.path.exists(f"models/{model_name}"):
        download_gcs_folder(source_path)

    model = DistilBertForSequenceClassification.from_pretrained(source_path, num_labels=2)
    model.to(device)
    return model

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict_string/") 
async def process_string(data: TextModel, model_name: str = "latest"):
    """
    Inference endpoint          
    """
    # check if model exists
    model = load_model(model_name = model_name)

    result = predict_string(model, data.text, device)

    # Extract prediction and probabilities from the result
    prediction = result["prediction"]
    probabilities = result["probabilities"]

    # Check if the prediction is 1 (human) or 0 (AI)
    prediction_label = "human" if prediction == 1 else "AI"

    # Get the probability for being human
    human_probability = probabilities[1] * 100  # Convert to percentage

    return f"This input is {prediction_label} with {human_probability:.2f}% probability"

@app.post("/process_csv/")
async def process_csv(file: UploadFile = File(...), model_name: str = "latest"):
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

    return FileResponse(local_predictions_file, media_type="text/csv", filename="predictions.csv")

# Instrumentator().instrument(app).expose(app)
