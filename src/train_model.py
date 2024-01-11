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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
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

    # Load the tokenizer
    # tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    #
    # # Function to tokenize the input texts
    # def tokenize_and_format(examples):
    #     # Tokenize the inputs and add the labels
    #     tokenized_inputs = tokenizer(examples['text'], padding='max_length', truncation=True)
    #     tokenized_inputs['labels'] = examples['generated']
    #     return tokenized_inputs
    #
    # # Load datasets
    # train_dataset = load_dataset('csv', data_files='data/processed/train.csv')['train']
    # test_dataset = load_dataset('csv', data_files='data/processed/test.csv')['train']
    #
    # # Tokenize datasets
    # train_dataset = train_dataset.map(tokenize_and_format, batched=True)
    # test_dataset = test_dataset.map(tokenize_and_format, batched=True)
    #
    # # Load the model
    # model = DistilBertForSequenceClassification.from_pretrained(parameters.model_args.model_type, num_labels=parameters.model_args.num_labels)
    # model.to(device)
    #
    # training_args = TrainingArguments(parameters.training_args)

    # Define training arguments       
    # training_args = TrainingArguments(
    #     output_dir='./results',
    #     num_train_epochs=parameters.num_train_epochs,
    #     per_device_train_batch_size=16,
    #     per_device_eval_batch_size=64,
    #     warmup_steps=500,
    #     weight_decay=0.01,
    #     logging_dir='./logs',
    #     logging_steps=10,
    # )

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

