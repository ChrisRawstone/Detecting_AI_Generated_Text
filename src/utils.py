import os
import pandas as pd
import numpy as np
from datasets import load_metric, load_from_disk
import wandb
import logging
import colorlog
from google.cloud import storage
from datetime import datetime
from src.visualizations.visualize import plot_confusion_matrix_sklearn

TEST_ROOT = os.path.dirname(__file__)  # root of test folder
PROJECT_ROOT = os.path.dirname(TEST_ROOT)

log = logging.getLogger(__name__)


def upload_to_gcs(local_model_dir: str, bucket_name: str, gcs_path: str, file_name: str, specific_file: str = ""):
    """
    Uploads files or a specific file from a local directory to Google Cloud Storage (GCS).

    Parameters:
    - local_model_dir (str): Local directory path containing the files to be uploaded.
    - bucket_name (str): The name of the GCS bucket to which the files will be uploaded.
    - gcs_path (str): The path within the GCS bucket where the files will be stored.
    - file_name (str): The name of the file or folder containing files to be uploaded.
    - specific_file (str, optional): If specified, only this file will be uploaded from the local directory.

    If `specific_file` is provided, only that file will be uploaded to GCS. If not provided, all files within the
    specified folder (`file_name`) in the local directory will be uploaded to GCS.

    Example:
    upload_to_gcs(
        local_model_dir="/path/to/local/directory",
        bucket_name="your-gcs-bucket",
        gcs_path="path/within/gcs/bucket",
        file_name="folder_name",
        specific_file="specific_file.txt"
    )
    """
    client = storage.Client()
    folder_name = f"{gcs_path}/{file_name}"
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(folder_name)
    if specific_file:
        local_file_path = os.path.join(local_model_dir, specific_file)
        blob.upload_from_filename(local_file_path)
    else:
        for local_file in os.listdir(os.path.join(local_model_dir, file_name)):
            local_file_path = os.path.join(local_model_dir, file_name, local_file)
            remote_blob_name = os.path.join(folder_name, local_file)
            # Upload the file to GCS
            blob = bucket.blob(remote_blob_name)
            blob.upload_from_filename(local_file_path)


def download_gcs_folder(source_folder: str, specific_file: str = "", bucket_name: str = "ai-detection-bucket"):
    """
    Downloads files from a Google Cloud Storage (GCS) folder to the local directory.

    Parameters:
    - source_folder (str): The path of the GCS folder to download from.
    - specific_file (str, optional): If specified, only this file will be downloaded from the GCS folder.
    - bucket_name (str, optional): The name of the GCS bucket. Default is "ai-detection-bucket".

    If `specific_file` is provided, only that file will be downloaded from the GCS folder. If not provided,
    all files within the specified GCS folder (`source_folder`) will be downloaded to the local directory.

    Example:
    download_gcs_folder(
        source_folder="path/within/gcs/bucket",
        specific_file="specific_file.txt",
        bucket_name="your-gcs-bucket"
    )
    """
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(bucket_name)

    if specific_file:
        blob = bucket.blob(os.path.join(source_folder, specific_file))
        os.makedirs(os.path.dirname(blob.name), exist_ok=True)
        blob.download_to_filename(blob.name)
    else:
        blobs = bucket.list_blobs(prefix=source_folder)  # Get list of files
        for blob in blobs:
            os.makedirs(os.path.dirname(blob.name), exist_ok=True)
            blob.download_to_filename(blob.name)


def load_model(model_name: str = "latest", source_folder: str = "models", device="cpu"):
    """
    Loads a pre-trained DistilBERT model for sequence classification.

    Parameters:
    - model_name (str, optional): The name of the model. Default is "latest".
    - source_folder (str, optional): The local or GCS path where the model is stored. Default is "models".
    - device (str, optional): The device on which to load the model (e.g., "cpu" or "cuda"). Default is "cpu".

    If the model is not found locally, it will attempt to download it from the specified GCS source folder.

    Returns:
    - model: The pre-trained DistilBERT model for sequence classification.

    Example:
    load_model(model_name="best_model", source_folder="path/to/model/folder", device="cuda")
    """
    from transformers import DistilBertForSequenceClassification

    source_path = os.path.join(source_folder, model_name)

    #create directory if not exists
    os.makedirs(f"models/{model_name}", exist_ok=True)
    download_gcs_folder(source_path)
    print("Downloaded model from GCS")

    model = DistilBertForSequenceClassification.from_pretrained(source_path, num_labels=2)
    model.to(device)
    return model


def load_csv(file_name: str = "train.csv", source_folder: str = "data/processed/csv_files/medium_data"):
    """
    Loads a CSV file into a Pandas DataFrame.

    Parameters:
    - file_name (str, optional): The name of the CSV file to load. Default is "train.csv".
    - source_folder (str, optional): The local or GCS path where the CSV file is stored. Default is
      "data/processed/csv_files/medium_data".

    If the CSV file is not found locally, it will attempt to download it from the specified GCS source folder.

    Returns:
    - pd.DataFrame: The Pandas DataFrame containing the data from the CSV file.

    Example:
    load_csv(file_name="test_data.csv", source_folder="path/to/csv/folder")
    """
    os.makedirs(source_folder, exist_ok=True)
    download_gcs_folder(source_folder, speficic_file=file_name)
    df = pd.read_csv(os.path.join(source_folder, file_name))
    return df

def download_latest_added_file(bucket_name: str="ai-detection-bucket",source_folder: str="inference_predictions"):
    # Initialize a client
    storage_client = storage.Client.create_anonymous_client()

    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=source_folder))  
    blob_names_list = [blob.name for blob in blobs]    
    blob_names_list = [name for name in blob_names_list if not name.endswith('/')]
    def get_timestamp(filename):
    # Extract the timestamp from the filename
        
        # remove csv extension
        filename = filename.split(".")[0]        
        timestamp_str = '_'.join(filename.split("_")[-2:])
        return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

    sorted_files= sorted(blob_names_list, key=get_timestamp, reverse=True)
    latest_file = sorted_files[0]
    # Download the file to a destination
    blob = bucket.blob(latest_file)
    os.makedirs(os.path.dirname(latest_file), exist_ok=True)
    blob.download_to_filename(latest_file)
    # return the path to the latest file

    return latest_file


def enable_wandb(parameters):
    """
    Initializes Weights and Biases (wandb) for experiment tracking if enabled.

    Parameters:
    - parameters: An object containing various parameters including 'general_args' and 'gcp_args'.
      Ensure 'wandb_enabled', 'model_name', and other necessary attributes are present.

    Returns:
    - bool: True if wandb is successfully initialized; False otherwise.

    Example:
    enable_wandb(parameters=my_parameters)
    """
    wandb_enabled = parameters.general_args.wandb_enabled
    if wandb_enabled == "True":
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

    return wandb_enabled


def compute_metrics(eval_pred):
    """
    Computes evaluation metrics for model predictions.

    Parameters:
    - eval_pred (tuple): A tuple containing model predictions' logits and corresponding labels.

    Returns:
    - dict: A dictionary containing computed evaluation metrics.

    The function uses the "accuracy" metric from the datasets library to compute accuracy.
    The computed metrics are logged, and the accuracy is returned in the dictionary.

    Example:
    compute_metrics(eval_pred=(logits, labels))
    """
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric.compute(predictions=predictions, references=labels)
    log.info(f"Accuracy: {accuracy['accuracy']}")
    return accuracy


def save_model(trainer, parameters):
    """
    Saves the trained model to the local directory and optionally uploads it to Google Cloud Storage (GCS).

    Parameters:
    - trainer: The Hugging Face Trainer object.
    - parameters: An object containing various parameters including 'gcp_args'.
      Ensure 'model_name', 'gcs_bucket', 'gcs_path', 'push_model_to_gcs', and other necessary attributes are present.

    The function saves the trained model in the local directory under '/models' with the specified model name.
    Additionally, it saves a copy of the model as 'latest'. If 'push_model_to_gcs' is set to "True" in parameters,
    the function uploads both the model with the specified name and 'latest' to the specified GCS bucket and path.

    Example:
    save_model(trainer=my_trainer, parameters=my_parameters)
    """
    model_dir = PROJECT_ROOT + "/models"
    trainer.save_model(f"{model_dir}/{parameters.gcp_args.model_name}")
    trainer.save_model(f"{model_dir}/latest")

    # Upload model to GCS
    if parameters.gcp_args.push_model_to_gcs == "True":
        upload_to_gcs(
            model_dir, parameters.gcp_args.gcs_bucket, parameters.gcp_args.gcs_path, parameters.gcp_args.model_name
        )
        upload_to_gcs(model_dir, parameters.gcp_args.gcs_bucket, parameters.gcp_args.gcs_path, "latest")


def wandb_log_metrics(all_predictions, class_names):
    """
    Logs various classification metrics to Weights and Biases (wandb).

    Parameters:
    - all_predictions (pd.DataFrame): DataFrame containing model predictions, including "prediction", "label", and "probabilities".
    - class_names (List[str]): List of class names for the confusion matrix and ROC curve.

    This function logs the following metrics to wandb:
    - Accuracy
    - Confusion Matrix
    - ROC Curve

    Additionally, it saves the confusion matrix plot to wandb using the `plot_confusion_matrix_sklearn` function.

    Example:
    wandb_log_metrics(all_predictions=my_predictions, class_names=["class_0", "class_1"])
    """
    metric = load_metric("accuracy")
    wandb.log(
        {
            "accuracy": metric.compute(predictions=all_predictions["prediction"], references=all_predictions["label"])[
                "accuracy"
            ]
        }
    )
    wandb.log(
        {
            "confusion matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_predictions["label"],
                preds=all_predictions["prediction"],
                class_names=class_names,
            )
        }
    )
    wandb.log(
        {
            "roc": wandb.plot.roc_curve(
                list(all_predictions["label"]), list(all_predictions["probabilities"]), labels=class_names
            )
        }
    )
    plot_confusion_matrix_sklearn(
        all_predictions["label"], all_predictions["prediction"], class_names, run=wandb.run
    )  # Saves to wandb

if __name__ == "__main__":
    model=load_model(model_name="experiment_1_GPU", device="cpu")