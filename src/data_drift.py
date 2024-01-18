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
from utils import upload_to_gcs, download_gcs_folder


def data_drift(reference_data: pd.DataFrame, current_data:pd.DataFrame, column_mapping: ColumnMapping):
    """
    Analyzes data drift between a reference dataset and a current dataset, generating three types of reports:
    1. Data Drift Report: Compares numerical and categorical distributions using Kolmogorov-Smirnov (KS) and Population
                          Stability Index (PSI) tests, with specified threshold values.
    2. Data Quality Report: Evaluates various data quality metrics for both reference and current datasets.
    3. Target Drift Report: Focuses on detecting drift in the target variable between reference and current datasets.

    Parameters:
    - reference_data (pd.DataFrame): The reference dataset used as a baseline for drift analysis.
    - current_data (pd.DataFrame): The current dataset for which data drift is being analyzed.
    - column_mapping (ColumnMapping): An object providing mapping information between columns in the datasets.

    Returns:
    Tuple[Report, Report, Report]: A tuple containing three reports:
    1. Data Drift Report.
    2. Data Quality Report.
    3. Target Drift Report.
    """
    data_drift_report = Report(
        metrics=[
            DataDriftPreset(
                num_stattest="ks", cat_stattest="psi", num_stattest_threshold=0.2, cat_stattest_threshold=0.2
            )
        ]
    )
    data_drift_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

    data_quality_report = Report(metrics=[DataQualityPreset()])
    data_quality_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    return data_drift_report, data_quality_report, target_drift_report


def save_reports(data_drift_report: Report, data_quality_report: Report, target_drift_report: Report, column_mapping: ColumnMapping):
    """
    Saves three data drift analysis reports to HTML files and uploads them to Google Cloud Storage (GCS).

    Parameters:
    - data_drift_report (Report): The data drift analysis report comparing numerical and categorical distributions.
    - data_quality_report (Report): The data quality analysis report evaluating various metrics for both datasets.
    - target_drift_report (Report): The target drift analysis report focusing on detecting drift in the target variable.
    - column_mapping (ColumnMapping): An object providing mapping information between columns in the datasets.

    The HTML files are saved to the local directory 'src/data_drifting' and are named:
    - 'report_data_drift_report.html'
    - 'report_data_quality_report.html'
    - 'report_target_drift_report.html'

    The function also uploads these HTML files to the specified Google Cloud Storage (GCS) bucket under the 'reports' path.

    GCS Upload Parameters:
    - bucket_name (str): The name of the GCS bucket to upload the reports.
    - gcs_path (str): The path within the GCS bucket where the reports will be stored.
    - file_names (List[str]): List of file names for the HTML reports in the local directory.

    Example:
    save_reports(data_drift_report, data_quality_report, target_drift_report, column_mapping)
    """

    base_directory = "src/data_drifting"
    os.makedirs(base_directory, exist_ok=True)

    reports = [data_drift_report, data_quality_report, target_drift_report]
    report_names = ["data_drift_report", "data_quality_report", "target_drift_report"]
    # Save each report to a separate HTML file
    for report, name in zip(reports, report_names):
        html_file_path = os.path.join(base_directory, f"report_{name}.html")
        report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
        report.save_html(html_file_path)
        upload_to_gcs(
            "src/data_drifting",
            bucket_name="ai-detection-bucket",
            gcs_path="reports",
            file_name=f"report_{name}.html",
            specific_file=f"report_{name}.html",
        )


def classification_report(reference_data: pd.DataFrame, current_data: pd.DataFrame, column_mapping: ColumnMapping):
    """
    Generates and saves a classification report comparing the target variable between reference and current datasets.

    Parameters:
    - reference_data (pd.DataFrame): The reference dataset used as a baseline for classification analysis.
    - current_data (pd.DataFrame): The current dataset for which classification analysis is being performed.
    - column_mapping (ColumnMapping): An object providing mapping information between columns in the datasets.

    The classification report includes various metrics for assessing the performance of a classification model,
    such as precision, recall, F1-score, and confusion matrix.

    The HTML report is saved to the local directory 'src/data_drifting' and is named:
    - 'report_classification_report.html'

    The function also uploads this HTML report to the specified Google Cloud Storage (GCS) bucket under the 'reports' path.

    GCS Upload Parameters:
    - bucket_name (str): The name of the GCS bucket to upload the report.
    - gcs_path (str): The path within the GCS bucket where the report will be stored.
    - gcs_file_name (str): The name for the HTML report in the GCS bucket.

    Example:
    classification_report(reference_data, current_data, column_mapping)
    """
    classification_report = Report(metrics=[ClassificationPreset()])
    classification_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    html_file_path = os.path.join("src/data_drifting", "report_classification_report.html")
    classification_report.save_html(html_file_path)
    upload_to_gcs(
        "src/data_drifting",
        bucket_name="ai-detection-bucket",
        gcs_path="reports",
        file_name="report_classification_report.html",
        specific_file="report_classification_report.html",
    )


if __name__ == "__main__":
    download_gcs_folder(
        source_folder="data/processed/csv_files/data_drift_files", specific_file="data_drift_essays.csv"
    )
    download_gcs_folder(source_folder="data/processed/csv_files/medium_data", specific_file="train.csv")
    download_gcs_folder(source_folder="inference_predictions", specific_file="predictions_20240117_170932.csv")

    reference_data = pd.read_csv("data/processed/csv_files/medium_data/train.csv")
    reference_data["prediction"] = reference_data["label"]

    current_data = pd.read_csv("inference_predictions/predictions_20240117_170932.csv")
    current_data["label"] = current_data["predictions"]
    current_data["prediction"] = current_data["predictions"]

    column_mapping = ColumnMapping(target="label", text_features=["text"])
    data_drift_report, data_quality_report, target_drift_report = data_drift(
        reference_data, current_data, column_mapping
    )
    save_reports(data_drift_report, data_quality_report, target_drift_report, column_mapping)

    column_mapping_classification = ColumnMapping(target="label", text_features=["text"], prediction="prediction")

    classification_report(reference_data, current_data, column_mapping_classification)
