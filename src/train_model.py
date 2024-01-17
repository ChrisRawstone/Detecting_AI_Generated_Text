import os
import numpy as np
import torch
from google.cloud import storage
from datasets import load_metric, load_from_disk
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from hydra.utils import get_original_cwd
import hydra
import omegaconf
from omegaconf import OmegaConf
import wandb
from src.predict_model import predict
from src.visualizations.visualize import plot_confusion_matrix_sklearn
import logging

# handler = colorlog.StreamHandler()
# handler.setFormatter(hydra_colorlog.ColoredFormatter(
# 	'%(log_color)s%(levelname)s:%(name)s:%(message)s'))

# hydra_logger = hydra.utils.log  # Use Hydra logger for logging

# hydra_logger.addHandler(handler)

log = logging.getLogger(__name__)


# Load metric for evaluation
metric = load_metric("accuracy")

TEST_ROOT = os.path.dirname(__file__)  # root of test folder
PROJECT_ROOT = os.path.dirname(TEST_ROOT)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric.compute(predictions=predictions, references=labels)
    log.info(f"Accuracy: {accuracy['accuracy']}")
    return accuracy

def upload_model_to_gcs(local_model_dir, bucket_name, gcs_path, model_name):
    client = storage.Client()
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
    log.info(f"Files uploaded to GCS folder: gs://{bucket_name}/{folder_name}")

def save_model(trainer, parameters): 
    model_dir = PROJECT_ROOT + "/models"
    trainer.save_model(f"{model_dir}/{parameters.gcp_args.model_name}")
    trainer.save_model(f"{model_dir}/latest")

    # Upload model to GCS
    if parameters.gcp_args.push_model_to_gcs == "True":
        upload_model_to_gcs(model_dir, parameters.gcp_args.gcs_bucket, parameters.gcp_args.gcs_path, parameters.gcp_args.model_name)
        upload_model_to_gcs(model_dir, parameters.gcp_args.gcs_bucket, parameters.gcp_args.gcs_path, "latest")

def wandb_log_metrics(all_predictions, class_names):
    wandb.log({"accuracy": metric.compute(predictions=all_predictions['prediction'], references=all_predictions['label'])['accuracy']})
    wandb.log({"confusion matrix": wandb.plot.confusion_matrix(
            probs=None, y_true=all_predictions['label'], preds=all_predictions['prediction'], class_names=class_names)})
    wandb.log({"roc": wandb.plot.roc_curve(list(all_predictions['label']), list(all_predictions['probabilities']), labels=class_names)})
    plot_confusion_matrix_sklearn(all_predictions['label'], all_predictions['prediction'], class_names, run=wandb.run)  # Saves to wandb

def train(config):
    print(omegaconf.OmegaConf.to_yaml(config))

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    log.info(f"Using device: {device}")

    parameters = config.experiment
    model = DistilBertForSequenceClassification.from_pretrained(parameters.model_settings.pretrained_model, num_labels=parameters.model_settings.num_labels)

    wandb_enabled = parameters.general_args.wandb_enabled
    if wandb_enabled == 'True':
        try:
            wandb.init(project="MLOps-DetectAIText", entity="teamdp", name=parameters.gcp_args.model_name)
            wandb_enabled = True
        except:
            print("Could not initialize wandb. No API key found.")
            wandb.init(mode="disabled")
            wandb_enabled = False
    else:
        wandb.init(mode="disabled")
        wandb_enabled = False

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
        compute_metrics=compute_metrics
        
        )

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    # Use the trained model for predictions
    model.eval()

    # Get predictions on the validation dataset
    all_predictions = predict(model, val_dataset, device)

    # Log metrics to wandb
    class_names = ["Human", "AI Generated"]
    if wandb_enabled:
        wandb_log_metrics(all_predictions, class_names)
       
    plot_confusion_matrix_sklearn(all_predictions['label'],all_predictions['prediction'],class_names,
                                  save_path=os.path.join(PROJECT_ROOT, "reports/figures"),
                                  name=f"confusion_matrix_{parameters.gcp_args.model_name}.png")  # Saves to reports/figures
    
    # Save the model
    save_model(trainer, parameters)
    
@hydra.main(config_path="config", config_name="default_config.yaml")
def main(config):
    train(config)

if __name__ == "__main__":
    main()
