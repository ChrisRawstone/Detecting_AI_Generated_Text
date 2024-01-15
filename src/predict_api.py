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

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict/") 
async def predict(text: str):
    """Inference endpoint          
    """
    result = predict_string(model,text,device)
    return result

 




# use this command to run the post request
# curl -X 'POST' "http://127.0.0.1:8000/predict/?text=some%20random%20text"



   