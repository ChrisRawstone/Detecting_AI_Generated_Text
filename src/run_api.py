from src.predict_model import predict_string
import torch
from fastapi import BackgroundTasks, FastAPI
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


app = FastAPI()



if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = DistilBertForSequenceClassification.from_pretrained(f"models/latest", num_labels=2)
model.to(device)

@app.post("/predict/")
def predict(text: str):
    """Inference endpoint 
    """
    return predict_string(model,text,device)   

    
    



   