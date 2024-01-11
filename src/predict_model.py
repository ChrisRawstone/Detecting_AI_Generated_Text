import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from data.make_dataset import tokenize_and_format
import pandas as pd
from datasets import load_from_disk
from torch.utils.data import DataLoader


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
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """

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

def predict_tokens(model: torch.nn.Module, tokenized_dataset: pd.DataFrame) -> None:
    """Run predictions for a given model on a dataframe that contains tokenized text
    
    Args:
        model: model to use for prediction
        dataframe: dataframe with a column 'text' containing the text to predict on
    
    Returns
        Tensor of shape [N] where N is the number of samples
    """

    # Create a DataLoader for the tokenized dataset
    dataloader = DataLoader(tokenized_dataset, batch_size=32)

    # Switch the model to evaluation mode
    model.eval()

    # Store predictions
    predictions = []

    # Disable gradient calculations for efficiency
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to the same device as model
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Get model outputs
            outputs = model(**batch)

            # Convert model outputs (logits) to probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

            # Get the predicted class (with the highest probability)
            predicted_classes = torch.argmax(probabilities, dim=1)

            # Add predictions to the list
            predictions.extend(predicted_classes.cpu().numpy())
    

    return predictions 


if __name__ == '__main__':

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    model = DistilBertForSequenceClassification.from_pretrained('models/latest', num_labels=2)
    model.to(device)

    tokenized_dataset = load_from_disk("data/processed/test_dataset_tokenized")

    predict_tokens(model, tokenized_dataset)





# Now, your 'test_csv' DataFrame will have an additional column 'predicted_label' with the predicted labels.





