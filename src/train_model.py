from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric, load_from_disk
import torch
import numpy as np
from accelerate import Accelerator  # Import the Accelerator class
import hydra
from hydra.utils import get_original_cwd 
import os

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric.compute(predictions=predictions, references=labels)
    print(f"Accuracy: {accuracy['accuracy']}") 
    return accuracy

@hydra.main(config_path="config", config_name="default_config.yaml",)
def main(config):    
    # Check if CUDA is available
    if torch.backends.mps.is_available():
        device = torch.device('mps') # For M1 Macs
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f'Using device: {device}')

    parameters = config.experiment

    model = DistilBertForSequenceClassification.from_pretrained(parameters.model_settings.cls, num_labels=parameters.model_settings.num_labels)

    # Load only 100 rows of data from the CSV files
    path_to_data = os.path.join(get_original_cwd(), 'data/processed')
    train_dataset = load_from_disk(os.path.join(path_to_data,"train_dataset_tokenized"))
    test_dataset = load_from_disk(os.path.join(path_to_data,"test_dataset_tokenized"))

    # Load the model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    model.to(device)

    training_args = TrainingArguments(**parameters.training_args)

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    trainer.save_model("models/latest")

if __name__ == '__main__':
    main()

