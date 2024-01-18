import os
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import DataQualityPreset
from evidently.metric_preset import ClassificationPreset
from evidently.metric_preset import TargetDriftPreset
from evidently.metrics import *
from evidently.tests import *
import nltk

nltk.download("words")
nltk.download("wordnet")
nltk.download("omw-1.4")

def download_data_from_gcs(local_download_dir, bucket_name, gcs_path, data_name):
    from google.cloud import storage
    import os
    client = storage.Client()
    folder_name = f"{gcs_path}/{data_name}"
    bucket = client.bucket(bucket_name)

    # Create local download directory if it doesn't exist
    os.makedirs(os.path.join(local_download_dir, data_name), exist_ok=True)

    # Download each file from the GCS folder to the local download directory
    blobs = bucket.list_blobs(prefix=folder_name)
    for blob in blobs:
        local_file_path = os.path.join(local_download_dir, data_name, os.path.basename(blob.name))
        blob.download_to_filename(local_file_path)

# Importing data
download_data_from_gcs(local_download_dir="Data_GCS", bucket_name="ai-detection-bucket", gcs_path="data", data_name="processed/csv_files")
download_data_from_gcs(local_download_dir="Data_GCS", bucket_name="ai-detection-bucket", gcs_path="data", data_name="processed/csv_files/medium_data")
download_data_from_gcs(local_download_dir="Data_GCS", bucket_name="ai-detection-bucket", gcs_path="data", data_name="inference_predictions/predictions_20240117_175237.csv")


def read_csv_files_from_directory(directory_path):
    if os.path.isdir(directory_path):
        files_in_directory = os.listdir(directory_path)
        csv_files = [os.path.join(directory_path, file) for file in files_in_directory if file.endswith('.csv')]
        for csv_file in csv_files:
            print(csv_file)
            data_frame = pd.read_csv(csv_file)
    return data_frame

# Read CSV files from the specified directories
reference_data = read_csv_files_from_directory("Data_GCS/processed/csv_files/medium_data/train.csv")
current_data = read_csv_files_from_directory("Data_GCS/processed/csv_files")
prediction_data = read_csv_files_from_directory("Data_GCS/inference_predictions/predictions_20240117_175237.csv")

current_data = read_csv_files_from_directory("Data_GCS/processed/csv_files")

reference_data["prediction"] = reference_data["label"]
current_data["prediction"] = current_data["label"]



# GLoVE embedding
column_mapping = ColumnMapping(target="generated", text_features=["text"], prediction="prediction")

# Generating reports
data_drift_report = Report(metrics=[DataDriftPreset(num_stattest="ks", cat_stattest="psi", num_stattest_threshold=0.2, cat_stattest_threshold=0.2)])
data_drift_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

data_quality_report = Report(metrics=[DataQualityPreset()])
data_quality_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

target_drift_report = Report(metrics=[TargetDriftPreset()])
target_drift_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

classification_report = Report(metrics=[ClassificationPreset()])
classification_report.run(reference_data=reference_data, current_data=prediction_data, column_mapping=column_mapping)

# Saving reports
base_directory = "src/data_drifting"
os.makedirs(base_directory, exist_ok=True)

reports = [data_drift_report, data_quality_report, target_drift_report, classification_report]
report_names = ["data_drift_report", "data_quality_report", "target_drift_report", "classification_report"]
# Save each report to a separate HTML file
for report, name in zip(reports, report_names):
    html_file_path = os.path.join(base_directory, f"report_{name}.html")
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    report.save_html(html_file_path)
