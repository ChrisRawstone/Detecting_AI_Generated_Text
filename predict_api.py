import os
from datetime import datetime

import pandas as pd
import torch
from fastapi import FastAPI, File, Request, UploadFile

# from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from google.cloud import storage
from pydantic import BaseModel

from src.predict_model import predict_csv, predict_string
from src.utils import load_model

app = FastAPI()
# bucket_name = "ai-detection-bucket"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


app.mount("/src/static", StaticFiles(directory="src/static"), name="src/static")


class TextModel(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    with open("src/static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict_string/")
async def process_string(data: TextModel, model_name: str = "experiment_1_GPU"):
    """
    Inference endpoint
    """
    # check if model exists
    model = load_model(model_name=model_name, device=device)

    result = predict_string(model, data.text, device)

    # Extract prediction and probabilities from the result
    prediction = result["prediction"]
    probabilities = result["probabilities"]

    # Check if the prediction is 1 (human) or 0 (AI)
    prediction_label = "human" if prediction == 0 else "AI"

    # Get the probability for being human
    human_probability = probabilities[0] * 100  # Convert to percentage

    if prediction_label == "human":
        probability = human_probability
    else:
        probability = 100 - human_probability

    return f"This input is {prediction_label} with {probability:.2f}% probability"


@app.post("/process_csv/")
async def process_csv(file: UploadFile = File(...), model_name: str = "experiment_1_GPU"):
    temp_file_path = "tempfile.csv"
    with open(temp_file_path, "wb") as buffer:
        content = await file.read()  # Read the file content
        buffer.write(content)  # Write to a temporary file

    # Read the CSV into a DataFrame
    df = pd.read_csv(temp_file_path)

    model = load_model(model_name=model_name, device=device)

    # Make predictions
    predictions_df = predict_csv(model, df, device)

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # make the results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    local_predictions_file = f"results/predictions_{timestamp}.csv"
    predictions_df.to_csv(local_predictions_file, index=False)

    # Upload to GCS # should probably be in a separate function but works
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket("ai-detection-bucket")
    gcs_file_path = f"inference_predictions/predictions_{timestamp}.csv"
    blob = bucket.blob(gcs_file_path)
    blob.upload_from_filename(local_predictions_file)

    return FileResponse(local_predictions_file, media_type="text/csv", filename="predictions.csv")
