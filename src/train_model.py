import numpy as np
from datetime import datetime as dt
import os
import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric, load_from_disk
import wandb
import hydra
from hydra.utils import get_original_cwd

hydra_logger = hydra.utils.log  # Use Hydra logger for logging

# Load metric for evaluation
metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric.compute(predictions=predictions, references=labels)
    hydra_logger.info(f"Accuracy: {accuracy['accuracy']}")
    return accuracy


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
            wandb.init(project="MLOps-DetectAIText", entity="teamdp", name=config.experiment.timestamp)
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
    trainer.save_model("model")
    trainer.save_model("../../latest")


if __name__ == "__main__":
    train()
