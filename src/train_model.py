import numpy as np
from datetime import datetime as dt
import os
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_metric, load_from_disk
import hydra
from hydra.utils import get_original_cwd
from google.cloud import storage
from src.visualizations.visualize import plot_confusion_matrix_sklearn
import wandb
import omegaconf
from hydra import compose, initialize
from omegaconf import OmegaConf

hydra_logger = hydra.utils.log  # Use Hydra logger for logging

# Load metric for evaluation
metric = load_metric("accuracy")

TEST_ROOT = os.path.dirname(__file__)  # root of test folder
PROJECT_ROOT = os.path.dirname(TEST_ROOT)


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
def main(config):
    train(config)


def train(config):
    print(omegaconf.OmegaConf.to_yaml(config))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    hydra_logger.info(f"Using device: {device}")

    parameters = config.experiment
    model = DistilBertForSequenceClassification.from_pretrained(
        parameters.model_settings.cls, num_labels=parameters.model_settings.num_labels
    )

    wandb_enabled = parameters.general_args.wandb_enabled
    if wandb_enabled:
        try:
            wandb.init(project="MLOps-DetectAIText", entity="teamdp", name=parameters.gcp_args.model_name)
        except:
            print("Could not initialize wandb. No API key found.")
            wandb.init(mode="disabled")
    else:
        wandb.init(mode="disabled")

    path_to_data = os.path.join(PROJECT_ROOT, "data/processed")
    train_dataset = load_from_disk(os.path.join(path_to_data, parameters.general_args.path_train_data))
    val_dataset = load_from_disk(os.path.join(path_to_data, parameters.general_args.path_val_data))
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

    # Use the trained model for predictions
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []
    prob1D = []
    for i in range(len(val_dataset)):
        with torch.no_grad():
            input_ids = torch.tensor(val_dataset[i]["input_ids"]).to(device)
            attention_mask = torch.tensor(val_dataset[i]["attention_mask"]).to(device)
            labels = val_dataset[i]["label"]  
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=1)
            predicted_label = torch.argmax(probs, dim=1).item()
            predictions.append(predicted_label)
            true_labels.append(labels)
            probabilities.append(list(probs.cpu().numpy().flatten()))
            prob1D.append(probs.cpu().numpy().flatten()[0])

    class_names = ["Human", "AI Generated"]
    # Log metrics to wandb
    wandb.log(
        {
            "confusion matrix": wandb.plot.confusion_matrix(
                probs=None, y_true=true_labels, preds=predictions, class_names=class_names
            )
        }
    )
    wandb.log({"roc": wandb.plot.roc_curve(true_labels, probabilities, labels=class_names)})
    plot_confusion_matrix_sklearn(true_labels, predictions, class_names, run=wandb.run) # Saves to wandb
    plot_confusion_matrix_sklearn(true_labels, predictions, class_names, save_path=os.path.join(PROJECT_ROOT, "reports/figures"), name=f"confusion_matrix_{parameters.gcp_args.model_name}.png") # Saves to reports/figures

    # Save the model
    model_dir = PROJECT_ROOT + "/models"
    trainer.save_model(f"{model_dir}/{parameters.gcp_args.model_name}")
    trainer.save_model(f"{model_dir}/latest")

    # Upload model to GCS
    if parameters.gcp_args.push_model_to_gcs == "True":
        upload_model_to_gcs(
            model_dir, parameters.gcp_args.gcs_bucket, parameters.gcp_args.gcs_path, parameters.gcp_args.model_name
        )
        upload_model_to_gcs(model_dir, parameters.gcp_args.gcs_bucket, parameters.gcp_args.gcs_path, "latest")


if __name__ == "__main__":
    main()
