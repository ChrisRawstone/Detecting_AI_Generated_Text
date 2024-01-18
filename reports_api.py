from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse
from google.cloud import storage
from src.predict_model import predict_string, predict_csv
from evidently import ColumnMapping

#from src.utils import download_gcs_folder, get_latest_added_file # we get this from data_drift
from src.utils import download_gcs_folder, download_latest_added_file
from src.data_drift import data_drift, save_reports, classification_report

import pandas as pd
import os

app = FastAPI()
bucket_name = "ai-detection-bucket"
reference_data_folder = "data/processed/csv_files/medium_data"
# download reference data also known as training data
download_gcs_folder(source_folder = reference_data_folder, specific_file = "train.csv")
# download the latest inference predictions
latest_file_name=download_latest_added_file()

@app.get("/")
async def root():
    return {"to get a report use the following endpoint": "/get_reports"}

@app.get("/get_reports")
async def get_reports(report_type: str = "data_drift"):
    """
    Data drift reports endpoint
    """

    reference_data = pd.read_csv(os.path.join(reference_data_folder, "train.csv"))
    reference_data["prediction"] = reference_data["label"]

    current_data = pd.read_csv(latest_file_name)
    current_data["label"] = current_data["prediction"]
      
    if report_type == "data_drift":      
        data_drift(reference_data, current_data)
        return {"message": "data drift reports generated"}        

    elif report_type == "classification":
        column_mapping_classification = ColumnMapping(target="label", text_features=["text"], prediction="prediction")
        classification_report(reference_data, current_data, column_mapping_classification)
        return {"message": "classification report generated"}
        
    else:
        return {"message": "report_type must be either data_drift or classification"}


 
    

