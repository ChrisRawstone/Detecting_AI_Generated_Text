import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
#from data.make_dataset import tokenize_and_format
import pandas as pd

def predict(
    model: torch.nn.Module,
    df: pd.DataFrame,
) -> None:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataframe: dataframe with a column 'text' containing the text to predict on
    
    Returns
        Tensor of shape [N] where N is the number of samples
    """

    predictions = []

    for index, row in df.iterrows():
        text = row['text'] 
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512, return_token_type_ids=False).to(device) # REMEMBER TO CHANGE TO
        
        with torch.no_grad():
            logits = model(**inputs).logits

        # Assuming it's binary classification (2 labels)
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()

        predictions.append(predicted_label)

    return predictions
 


    
if __name__ == '__main__':

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    model = DistilBertForSequenceClassification.from_pretrained('model/latest', num_labels=2)
    model.to(device)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    test_csv = pd.read_csv('data/processed/test.csv')

    predictions = predict(model, test_csv)

    print(predictions)




# Now, your 'test_csv' DataFrame will have an additional column 'predicted_label' with the predicted labels.





