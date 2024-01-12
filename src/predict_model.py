import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from src.data.make_dataset import tokenize_and_format
import pandas as pd
import datasets
from datasets import load_from_disk
from torch.utils.data import DataLoader
import os
from datetime import datetime

def predict_dataframe(
    model: torch.nn.Module,
    df: pd.DataFrame    
) -> None:
    """Run predictions for a given model on a dataframe. Non-tokenized!
    
    Args:
        model: model to use for prediction
        dataframe: dataframe with a column 'text' containing the text to predict on
        probabilities: boolean flag whether to return 
    
    Returns
        Tensor of shape [N] where N is the number of samples
    """

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    predictions = []

    for index, row in df.iterrows():
        text = row['text'] 
        inputs = tokenizer(text, 
                           return_tensors='pt', 
                           truncation=True, 
                           padding=True, 
                           max_length=512, 
                           return_token_type_ids=False).to(device) 

        with torch.no_grad():
            logits = model(**inputs).logits
    
        # Assuming it's binary classification (2 labels)
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()

        predictions.append(predicted_label)

    return predictions

def predict_tokens(model: torch.nn.Module, tokenized_dataset : datasets.arrow_dataset.Dataset, device) -> None:
    """Run predictions for a given model on a dataframe that contains tokenized text
    
    Args:
        model: model to use for prediction
        dataframe: dataframe with a column 'text' containing the text to predict on
    
    Returns
        Dataframe
    """

    predictions_dataframe = pd.DataFrame(columns=['text'])

    predictions = []

    for i in range(len(tokenized_dataset)):
        with torch.no_grad():
            input_ids = torch.tensor(tokenized_dataset[i]['input_ids']).to(device)  # Convert input_ids to a tensor
            attention_mask = torch.tensor(tokenized_dataset[i]['attention_mask']).to(device)  # Convert attention_mask to a tensor

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            probabilities = torch.softmax(logits, dim=1)
            predicted_label = torch.argmax(probabilities, dim=1).item()

            predictions.append(predicted_label)
    
    predictions_dataframe['text'] = tokenized_dataset['text']
    predictions_dataframe['prediction'] = predictions
    predictions_dataframe['generated'] = tokenized_dataset['generated']
    return predictions_dataframe 

def find_latest_folder(path):
    # Get a list of all date folders in the given path
    date_folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d != "latest"]

    if not date_folders:
        print("No date folders found.")
        return None

    # Find the folder with the latest date
    latest_date_folder = max(date_folders)

    # Construct the full path to the latest date folder
    latest_date_folder_path = os.path.join(path, latest_date_folder)

    # Get a list of all time folders within the latest date folder
    time_folders = [t for t in os.listdir(latest_date_folder_path) if os.path.isdir(os.path.join(latest_date_folder_path, t))]

    if not time_folders:
        print("No time folders found in the latest date folder.")
        return None

    # Find the folder with the latest time
    latest_time_folder = max(time_folders)

    # Construct the full path to the latest time folder
    model_name = latest_date_folder + "-" + latest_time_folder

    return model_name

if __name__ == '__main__':

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model_name = "latest"

    model = DistilBertForSequenceClassification.from_pretrained(f'models/{model_name}', num_labels=2)
    model.to(device)

    tokenized_dataset = load_from_disk("data/processed/test_dataset_tokenized")

    predictions_df = predict_tokens(model, tokenized_dataset, device)

    predictions_df.to_json(f"results/{model_name}.json", orient='table', indent=1)








