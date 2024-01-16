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
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Tokenize the data
    tokenized_text = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt', max_length=512).to(device)     

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

    for i in range(len(tokenized_dataset)):
        with torch.no_grad():
            input_ids = torch.tensor(tokenized_dataset[i]["input_ids"]).to(device)  # Convert input_ids to a tensor
            attention_mask = torch.tensor(tokenized_dataset[i]["attention_mask"]).to(
                device
            )  # Convert attention_mask to a tensor

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            probabilities = torch.softmax(logits, dim=1)

            predicted_label = torch.argmax(probabilities, dim=1).item()

            predictions.append(predicted_label)

    predictions_dataframe["text"] = tokenized_dataset["text"]
    predictions_dataframe["prediction"] = predictions
    predictions_dataframe["generated"] = tokenized_dataset["generated"]
    return predictions_dataframe

if __name__ == "__main__":
    # print pwd
    import os

    print(os.getcwd())

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = DistilBertForSequenceClassification.from_pretrained(f"models/latest", num_labels=2)
    model.to(device)

    tokenized_dataset = load_from_disk("data/processed/test_dataset_tokenized")

    predictions_df = predict(model, tokenized_dataset, device)

    print("Predictions:\n", predictions_df.head(5))

    model_name = 'debug'

    if not os.path.exists("results"):
        os.makedirs("results")

    predictions_df.to_json(f"results/predictions_{model_name}.json", orient="table", indent=1)
    print("Predictions saved to results folder")
