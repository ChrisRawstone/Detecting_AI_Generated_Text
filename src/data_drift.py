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
from src.utils import upload_to_gcs, download_gcs_folder

def data_drift(reference_data, current_data, column_mapping):
    # Generating reports
    data_drift_report = Report(metrics=[DataDriftPreset(num_stattest="ks", cat_stattest="psi", num_stattest_threshold=0.2, cat_stattest_threshold=0.2)])
    data_drift_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

    data_quality_report = Report(metrics=[DataQualityPreset()])
    data_quality_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    return data_drift_report, data_quality_report, target_drift_report

def save_reports(data_drift_report, data_quality_report, target_drift_report, column_mapping):
    # Saving reports
    base_directory = "src/data_drifting"
    os.makedirs(base_directory, exist_ok=True)

    reports = [data_drift_report, data_quality_report, target_drift_report]
    report_names = ["data_drift_report", "data_quality_report", "target_drift_report"]
    # Save each report to a separate HTML file
    for report, name in zip(reports, report_names):
        html_file_path = os.path.join(base_directory, f"report_{name}.html")
        report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
        report.save_html(html_file_path)
        upload_to_gcs('src/data_drifting', bucket_name='ai-detection-bucket', gcs_path='reports', file_name=f"report_{name}.html", specific_file=f"report_{name}.html")

def classification_report(reference_data, current_data, column_mapping):
    classification_report = Report(metrics=[ClassificationPreset()])
    classification_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    html_file_path = os.path.join("src/data_drifting", "report_classification_report.html")
    classification_report.save_html(html_file_path)
    upload_to_gcs('src/data_drifting', bucket_name='ai-detection-bucket', gcs_path='reports', file_name="report_classification_report.html", specific_file="report_classification_report.html")

if __name__ == "__main__":
    download_gcs_folder(source_folder = "data/processed/csv_files/data_drift_files", specific_file = "data_drift_essays.csv")
    download_gcs_folder(source_folder = "data/processed/csv_files/medium_data", specific_file = "train.csv")
    download_gcs_folder(source_folder = "inference_predictions", specific_file = "predictions_20240117_170932.csv")
    
    reference_data = pd.read_csv("data/processed/csv_files/medium_data/train.csv")
    reference_data["prediction"] = reference_data["label"]

    current_data = pd.read_csv("inference_predictions/predictions_20240117_170932.csv")
    current_data["label"] = current_data["prediction"]
            
    column_mapping = ColumnMapping(target="label", text_features=["text"])
    data_drift_report, data_quality_report, target_drift_report = data_drift(reference_data, current_data, column_mapping)
    save_reports(data_drift_report, data_quality_report, target_drift_report, column_mapping)
    
    column_mapping_classification = ColumnMapping(target="label", text_features=["text"], prediction="prediction")

    classification_report(reference_data, current_data, column_mapping_classification)
    