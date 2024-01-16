import numpy as np
from datetime import datetime as dt
import os
import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_metric, load_from_disk
import hydra
from hydra.utils import get_original_cwd
from google.cloud import storage
import wandb

hydra_logger = hydra.utils.log  # Use Hydra logger for logging

# Load metric for evaluation
metric = load_metric("accuracy")

import os
wandb_api_key = os.environ.get('WANDB_API_KEY')
print("WANDB_API_KEY: ", wandb_api_key)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric.compute(predictions=predictions, references=labels)
    hydra_logger.info(f"Accuracy: {accuracy['accuracy']}")
    return accuracy

def upload_model_to_gcs(local_model_dir, bucket_name, gcs_path, model_name):
    client = storage.Client()

    # Create the folder in GCS
    folder_name = f"{gcs_path}/{model_name}"
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(folder_name)

    # Upload each file from local_model_dir to the GCS folder
    for local_file in os.listdir(os.path.join(local_model_dir, model_name)):
        local_file_path = os.path.join(local_model_dir, model_name, local_file)
        remote_blob_name = os.path.join(folder_name, local_file)
        # Upload the file to GCS
        blob = bucket.blob(remote_blob_name)
        blob.upload_from_filename(local_file_path)

    hydra_logger.info(f"Files uploaded to GCS folder: gs://{bucket_name}/{folder_name}")


@hydra.main(config_path="config", config_name="default_config.yaml")
def train(config):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    hydra_logger.info(f"Using device: {device}")

    parameters = config.experiment
    model = DistilBertForSequenceClassification.from_pretrained(
        parameters.model_settings.cls, num_labels=parameters.model_settings.num_labels
    )

    wandb_enabled = True
    if wandb_enabled:
        
        try:
            wandb.login(key=wandb_api_key)
            wandb.init(project="MLOps-DetectAIText", entity="teamdp", name=parameters.gcp_args.model_name)
        except:
            print("Could not initialize wandb. No API key found.")
            wandb.init(mode="disabled")
    else:
        wandb.init(mode="disabled")

    path_to_data = os.path.join(get_original_cwd(), "data/processed")
    train_dataset = load_from_disk(os.path.join(path_to_data, "train_dataset_tokenized"))
    val_dataset = load_from_disk(os.path.join(path_to_data, "val_dataset_tokenized"))
    hydra_logger.info(f"Length of train data: {(len(train_dataset))}")

    # Load the model
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.to(device)

    training_args = TrainingArguments(**parameters.training_args)

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    # Save the model
    model_dir = '../../../models'
    trainer.save_model(f"{model_dir}/{parameters.gcp_args.model_name}")
    trainer.save_model(f"{model_dir}/latest")

    # Upload model to GCS
    upload_model_to_gcs(model_dir, parameters.gcp_args.gcs_bucket, parameters.gcp_args.gcs_path, parameters.gcp_args.model_name)
    upload_model_to_gcs(model_dir, parameters.gcp_args.gcs_bucket, parameters.gcp_args.gcs_path, "latest")

if __name__ == "__main__":
    train()
