import os
import pandas as pd
import datasets
import torch
from transformers import DistilBertForSequenceClassification
from datasets import load_from_disk
from typing import Dict
from transformers import DistilBertTokenizerFast

def predict_string(model: torch.nn.Module, text: str, device) -> Dict:
    """Run predictions for a given model on a string

    Args:
        model: model to use for prediction
        text: text to predict on

    Returns
        Dict
    """
    # Tokenize the text
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Tokenize the data
    tokenized_text = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt", max_length=512).to(device)
    
    # Get the model prediction
    with torch.no_grad():
        logits = model(**tokenized_text).logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()     
    
    return {
        'text': text,
        'prediction': prediction,
        'probabilities': probabilities.tolist()[0]
    }  

def predict_csv(model: torch.nn.Module, dataframe: pd.DataFrame, device, batch_size: int=32) -> pd.DataFrame: 
    """Run predictions for a given model on a csv file
    
    Args:
        model: model to use for prediction
        csv: pandas dataframe with a column 'text' containing the text to predict on 
        device: device to use for prediction
        tokenizer: pre-loaded tokenizer.
        batch_size: size of the batches for prediction
    
    Returns
        pd.DataFrame with predictions
    """
    # Tokenize the text
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Tokenize the text in batches
    def tokenize_batch(batch):
        return tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512)

    # Process in batches
    predictions = []
    for i in range(0, len(dataframe), batch_size):
        batch = dataframe['text'][i:i+batch_size].tolist()
        tokenized = tokenize_batch(batch)
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        with torch.no_grad():
            logits = model(**tokenized).logits
            probabilities = torch.softmax(logits, dim=1)
            batch_predictions = torch.argmax(probabilities, dim=1).tolist()
            predictions.extend(batch_predictions)

    dataframe['predictions'] = predictions

    return dataframe
    

def predict(model: torch.nn.Module, tokenized_dataset: datasets.arrow_dataset.Dataset, device) -> None:
    """Run predictions for a given model on a dataframe that contains tokenized text

    Args:
        model: model to use for prediction
        dataframe: dataframe with a column 'text' containing the text to predict on

    Returns
        Dataframe
    """

    predictions_dataframe = pd.DataFrame(columns=["text"])

    predictions = []
    probabilities = []
    for i in range(len(tokenized_dataset)):
        with torch.no_grad():
            input_ids = torch.tensor(tokenized_dataset[i]["input_ids"]).to(device)  # Convert input_ids to a tensor
            attention_mask = torch.tensor(tokenized_dataset[i]["attention_mask"]).to(device)  # Convert attention_mask to a tensor

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            probs = torch.softmax(logits, dim=1)

            predicted_label = torch.argmax(probs, dim=1).item()

            predictions.append(predicted_label)
            probabilities.append(list(probs.cpu().numpy().flatten()))

    predictions_dataframe["text"] = tokenized_dataset["text"]
    predictions_dataframe["prediction"] = predictions
    predictions_dataframe["probabilities"] = probabilities
    predictions_dataframe["label"] = tokenized_dataset["label"]
    return predictions_dataframe

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = DistilBertForSequenceClassification.from_pretrained(f"models/latest", num_labels=2)
    model.to(device)

    # load the test dataset
    test_df = pd.read_csv("data/processed/csv_files/small_data/test.csv")

    predict_csv(model, test_df, device, batch_size=32)

    tokenized_dataset = load_from_disk("data/processed/tokenized_data/small_data/test_dataset_tokenized")

    predictions_df = predict(model, tokenized_dataset, device)


    print("Predictions:\n", predictions_df.head(5))

    model_name = "debug"

    if not os.path.exists("results"):
        os.makedirs("results")

    predictions_df.to_json(f"results/predictions_{model_name}.json", orient="table", indent=1)
    print("Predictions saved to results folder")
