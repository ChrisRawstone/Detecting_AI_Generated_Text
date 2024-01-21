import os

import click
import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast


@click.group()
def cli():
    """Command line interface."""
    pass


def get_data_original_data(path: str):
    """This function gets the original data from the raw folder
    Args:
        path (string): Path to the raw folder
    Returns:
        pd.DataFrame: Original essays
    """

    # Get the original essays
    if not os.path.exists(path):
        raise FileNotFoundError("No generated data found. Please run DVC pull.")
    original_essays = pd.read_csv(path + "original_data/train_essays.csv")
    original_essays.drop(columns=["id"], inplace=True)

    # Rename the columns
    original_essays = original_essays.rename(columns={"generated": "label"})

    return original_essays


def get_data_generated_data_first_iteration(path: str):
    """This function gets the generated data from the raw folder from the first iteration of data generated by our own prompt script
    Args:
        path (string): Path to the raw folder
    Returns:
        pd.DataFrame: Generated essays
    """

    generated_essays = pd.DataFrame()
    if not os.path.exists(path):
        raise FileNotFoundError("No generated data found. Please run DVC pull.")
    for i in range(1, 6):
        generated_essays = pd.concat(
            [generated_essays, pd.read_csv(path + "generated_data/AI_Generated_df{}.csv".format(i))]
        )
    generated_essays = generated_essays.rename(columns={"generated_text": "text", "generated": "label"})

    return generated_essays[["text", "label"]]


def get_DAIGTProperTrainDataset(path: str):
    """This function gets the generated data from the raw folder from the DAIGT DATASET
    Args:
        path (string): Path to the raw folder
    Returns:
        pd.DataFrame: Generated essays
    """

    DAIGTProperTrainDataset_essays = pd.DataFrame()
    if not os.path.exists(path):
        raise FileNotFoundError("No generated data found. Please run DVC pull.")
    for i in range(1, 5):
        DAIGTProperTrainDataset_essays = pd.concat(
            [
                DAIGTProperTrainDataset_essays,
                pd.read_csv(path + "DAIGTProperTrainDataset/train_drcat_0{}.csv".format(i)),
            ]
        )

    return DAIGTProperTrainDataset_essays[["text", "label"]]


def get_AugmenteddataforLLM(path: str):
    """This function gets the generated data from the raw folder from the AugmenteddataforLLM DATASET
    Args:
        path (string): Path to the raw folder
    Returns:
        pd.DataFrame: Generated essays
    """

    AugmenteddataforLLM_essays = pd.DataFrame()
    if not os.path.exists(path):
        raise FileNotFoundError("No generated data found. Please run DVC pull.")
    for name in ["final_test.csv", "final_train.csv"]:
        AugmenteddataforLLM_essays = pd.concat(
            [AugmenteddataforLLM_essays, pd.read_csv(path + f"AugmenteddataforLLM/{name}")]
        )
    return AugmenteddataforLLM_essays[["text", "label"]]

def DAIGTv4(path: str):
    """This function gets the generated data from the raw folder from the DAIGHTv4 DATASET
    Args:
        path (string): Path to the raw folder
    Returns:
        pd.DataFrame: Generated essays
    """



    DAIGTv4_essays = pd.DataFrame()
    if not os.path.exists(path):
        raise FileNotFoundError("No generated data found. Please run DVC pull.")
    for name in ["daigt_magic_generations.csv", "train_v4_drcat_01.csv"]:
        DAIGTv4_essays = pd.concat(
            [DAIGTv4_essays, pd.read_csv(path + f"DAIGTv4/{name}")]
        )
    return DAIGTv4_essays[["text", "label"]]

def DAIGTv2(path: str):
    """This function gets the generated data from the raw folder from the DAIGHTv2 DATASET
    Args:
        path (string): Path to the raw folder
    Returns:
        pd.DataFrame: Generated essays
    """


    DAIGTv2_essays = pd.DataFrame()
    if not os.path.exists(path):
        raise FileNotFoundError("No generated data found. Please run DVC pull.")
    for name in ["train_v2_drcat_02.csv"]:
        DAIGTv2_essays = pd.concat(
            [DAIGTv2_essays, pd.read_csv(path + f"DAIGTv2/{name}")]
        )
    return DAIGTv2_essays[["text", "label"]]

def get_daigt_extended(path: str):
    """This function gets the generated data from the raw folder from the DAIGHTv2 DATASET
    Args:
        path (string): Path to the raw folder
    Returns:
        pd.DataFrame: Generated essays
    """


    daigt_extended = pd.DataFrame()
    if not os.path.exists(path):
        raise FileNotFoundError("No generated data found. Please run DVC pull.")
    for name in ["concatenated.csv"]:
        daigt_extended = pd.concat(
            [daigt_extended, pd.read_csv(path + f"Daigt_extended/{name}")]
        )
    daigt_extended = daigt_extended.rename(columns={"generated": "label"})
    
    return daigt_extended[["text", "label"]]


def get_data(sample_size: int = None, path: str = "data/raw/"):
    """This function gets the generated and human data from the raw folder, concats them and splits them into train and test
    Args:
        sample_size (int, optional): Sample size of the data. Defaults to None.

    Returns:
        X_train: train data
        X_test: test data
    """

    original_essays = get_data_original_data(path)
    generated_essays = get_data_generated_data_first_iteration(path)
    # DAIGTProperTrainDataset_essays = get_DAIGTProperTrainDataset(path)
    AugmenteddataforLLM_essays = get_AugmenteddataforLLM(path)
    DAIGTv2_essays = DAIGTv2(path)
    DAIGTv4_essays = DAIGTv4(path)
    daigt_extended = get_daigt_extended(path)


    # Concat all the dataframes
    all_essays = pd.concat([original_essays, generated_essays])
    # all_essays = pd.concat([all_essays, DAIGTProperTrainDataset_essays])
    all_essays = pd.concat([all_essays, AugmenteddataforLLM_essays])
    all_essays = pd.concat([all_essays, DAIGTv2_essays])
    all_essays = pd.concat([all_essays, daigt_extended])

    # Reset the index
    all_essays.reset_index(inplace=True, drop=True)

    # create key from index
    all_essays["key"] = all_essays.index
    all_essays["label"] = all_essays["label"].astype(int)



    # Sample the data if needed
    if sample_size:
        all_essays = all_essays.sample(sample_size, random_state=42)

    # Split the data into train and test with sklearn
    X_train, X_temp = train_test_split(all_essays[["text", "label"]], test_size=0.2, random_state=42)
    X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

    return X_train, X_test, X_val


def tokenize_and_format(data: Dataset):
    """This function tokenizes the data and formats it for the model

    Args:
        data load_dataset.Dataset: Data to be tokenized

    Returns:
        load_dataset.Dataset: Tokenized data
    """
    # Load the tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Tokenize the data
    tokenized_inputs = tokenizer(
        data["text"], padding="max_length", truncation=True, return_tensors="pt", max_length=512
    )
    tokenized_inputs["label"] = data["label"]

    return tokenized_inputs


@click.command()
@click.option("--sample_size", type=int, default=None, help="sample size")
def make_dataset(sample_size):
    """Makes the dataset by getting the data, tokenizing it and saving it to disk

    Args:
        sample_size (int, optional): Sample size of the data. Defaults to None.
    """

    # Get the data
    X_train, X_test, X_val = get_data(sample_size)

    # Load datasets
    train_dataset = Dataset.from_pandas(X_train)
    val_dataset = Dataset.from_pandas(X_val)
    test_dataset = Dataset.from_pandas(X_test)

    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_and_format, batched=True)
    val_dataset = val_dataset.map(tokenize_and_format, batched=True)
    test_dataset = test_dataset.map(tokenize_and_format, batched=True)

    if sample_size is None:
        X_train.to_csv("data/processed/csv_files/full_data/train.csv", index=False)
        X_test.to_csv("data/processed/csv_files/full_data/test.csv", index=False)
        X_val.to_csv("data/processed/csv_files/full_data/validation.csv", index=False)

        train_dataset.save_to_disk("data/processed/tokenized_data/full_data/train_dataset_tokenized")
        val_dataset.save_to_disk("data/processed/tokenized_data/full_data/val_dataset_tokenized")
        test_dataset.save_to_disk("data/processed/tokenized_data/full_data/test_dataset_tokenized")


    else:


        X_train.to_csv("data/processed/csv_files/small_data/train.csv", index=False)
        X_test.to_csv("data/processed/csv_files/small_data/test.csv", index=False)
        X_val.to_csv("data/processed/csv_files/small_data/validation.csv", index=False)



if __name__ == "__main__":
    # Get the data and process it
    make_dataset()
