import os
import torch
from datasets import load_metric, load_from_disk
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
import hydra
import omegaconf
from src.predict_model import predict
import logging
from transformers.utils import logging as transformer_logging
from src.utils import upload_to_gcs, save_model, enable_wandb, wandb_log_metrics, compute_metrics

log = logging.getLogger(__name__)

logger = transformer_logging.get_logger("transformers")

logger.setLevel(transformer_logging.WARNING)

# Load metric for evaluation
metric = load_metric("accuracy")

TEST_ROOT = os.path.dirname(__file__)  # root of test folder
PROJECT_ROOT = os.path.dirname(TEST_ROOT)


def train(config):
    print(omegaconf.OmegaConf.to_yaml(config))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    log.info(f"Using device: {device}")

    parameters = config.experiment
    model = DistilBertForSequenceClassification.from_pretrained(
        parameters.model_settings.pretrained_model, num_labels=parameters.model_settings.num_labels
    )

    wandb_enabled = enable_wandb(parameters)

    path_to_data = os.path.join(PROJECT_ROOT, "data/processed")
    train_dataset = load_from_disk(os.path.join(path_to_data, parameters.general_args.path_train_data))
    val_dataset = load_from_disk(os.path.join(path_to_data, parameters.general_args.path_val_data))
    log.info(f"Length of train data: {(len(train_dataset))}")

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

    # Save the model
    save_model(trainer, parameters)

    # Get predictions on the validation dataset
    all_predictions = predict(model, val_dataset, device)

    # Log metrics to wandb
    class_names = ["Human", "AI Generated"]
    if wandb_enabled:
        wandb_log_metrics(all_predictions, class_names)


@hydra.main(config_path="config", config_name="default_config.yaml")
def main(config):
    train(config)


if __name__ == "__main__":
    main()
