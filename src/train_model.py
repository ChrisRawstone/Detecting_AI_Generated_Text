from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric, load_from_disk
import torch
import numpy as np
from accelerate import Accelerator  # Import the Accelerator class
# import hydra

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric.compute(predictions=predictions, references=labels)
    print(f"Accuracy: {accuracy['accuracy']}")
    return accuracy




def main():
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f'Using device: {device}')



    # Load only 100 rows of data from the CSV files
    train_dataset = load_from_disk('data/processed/train_dataset_tokenized')
    test_dataset = load_from_disk('data/processed/test_dataset_tokenized')



    # Load the model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    model.to(device)

    # Define training arguments
    #@hydre
   
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
    )

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



if __name__ == '__main__':
    main()

