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
nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Importing data
reference_data = pd.read_csv('data/processed/train.csv')
reference_data['prediction'] = reference_data['generated']
current_data = pd.read_csv('data/processed/test.csv')
current_data['prediction'] = current_data['generated']
prediction_data = pd.read_json('results/predictions_debug.json', orient='table')

# GLoVE embedding
column_mapping = ColumnMapping(
    target='generated',
    text_features=['text'], 
    prediction='prediction')

# Generating reports
data_drift_report = Report(metrics=[
    DataDriftPreset(num_stattest='ks', cat_stattest='psi', num_stattest_threshold=0.2, cat_stattest_threshold=0.2),])
data_drift_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

data_quality_report = Report(metrics=[
    DataQualityPreset()])
data_quality_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

target_drift_report = Report(metrics=[
    TargetDriftPreset()])
target_drift_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

classification_report = Report(metrics=[
    ClassificationPreset()])
classification_report.run(reference_data=reference_data, current_data=prediction_data, column_mapping=column_mapping)
    
# Saving reports
base_directory = 'src/data_drifting'
os.makedirs(base_directory, exist_ok=True)

reports = [data_drift_report, data_quality_report, target_drift_report, classification_report]
report_names = ['data_drift_report','data_quality_report','target_drift_report', 'classification_report']
# Save each report to a separate HTML file
for report, name in zip(reports, report_names):
    html_file_path = os.path.join(base_directory, f'report_{name}.html')
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    report.save_html(html_file_path)