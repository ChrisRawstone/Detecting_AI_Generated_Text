import sys
import pytest
import os.path
import pandas as pd
import torch
import json
from mypaths import PROJECT_ROOT
sys.path.append(PROJECT_ROOT)
from src.predict_model import predict,find_latest_folder
from datasets import load_from_disk
from transformers import DistilBertForSequenceClassification

model_name = find_latest_folder('models')
@pytest.mark.skipif(not os.path.exists('results/predictions_{model_name}.json'), reason="Required data files not found")
def test_predict_model():
    # Read the predictions from json file
    with open('results/predictions', 'r') as file:
        predictions_dataframe = json.load(file)
    
    assert 'text' in predictions_dataframe.columns
    assert 'prediction' in predictions_dataframe.columns
    assert 'generated' in predictions_dataframe.columns

    assert len(predictions_dataframe) >= 1    
    assert set(predictions_dataframe['prediction'].unique()) == {0,1}    
    assert set(predictions_dataframe['generated'].unique()) == {0,1}

