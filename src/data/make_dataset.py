import pandas as pd 
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
from datasets import load_dataset, Dataset
import click

@click.group()
def cli():
    """Command line interface."""
    pass

def get_data(sample_size: int = None):
    """This function gets the generated and human data from the raw folder, concats them and splits them into train and test
    Args:
        sample_size (int, optional): Sample size of the data. Defaults to None.
    
    Returns:
        X_train: train data
        X_test: test data
    """
    # Concat all generated essays
    generated_essays = pd.DataFrame()
    for i in range(1,6):
        generated_essays = pd.concat([generated_essays, pd.read_csv("data/raw/generated_data/AI_Generated_df{}.csv".format(i))])
    generated_essays = generated_essays.rename(columns={"generated_text":"text"})

    # Get the original essays
    original_essays = pd.read_csv("data/raw/train_essays.csv")
    original_essays.drop(columns=["id"], inplace=True)

    # Concat the two dataframes
    df = pd.concat([original_essays, generated_essays])
    df.reset_index(inplace=True, drop=True)
    # create key from index
    df["key"] = df.index
    df['generated'] = df['generated'].astype(int)

    # Sample the data if needed
    if sample_size:
        df = df.sample(sample_size, random_state=42)

    # Split the data into train and test with sklearn
    X_train, X_test = train_test_split(df[['text','generated']], test_size=0.2, random_state=42)

    X_train.to_csv("data/processed/train.csv", index=False)
    X_test.to_csv("data/processed/test.csv", index=False)

def tokenize_and_format(data: Dataset):
    """This function tokenizes the data and formats it for the model

    Args:
        data load_dataset.Dataset: Data to be tokenized

    Returns:
        load_dataset.Dataset: Tokenized data
    """
    # Load the tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Tokenize the data
    tokenized_inputs = tokenizer(data['text'], padding='max_length', truncation=True)
    tokenized_inputs['labels'] = data['generated']
    
    return tokenized_inputs


@click.command()
@click.option("--sample_size", type=int, default=None, help="sample size")
def make_dataset(sample_size):
    """Makes the dataset by getting the data, tokenizing it and saving it to disk
    
    Args:
        sample_size (int, optional): Sample size of the data. Defaults to None.
    """

    # Get the data
    get_data(sample_size)

    # Load datasets
    train_dataset = load_dataset('csv', data_files='data/processed/train.csv')['train']
    test_dataset = load_dataset('csv', data_files='data/processed/test.csv')['train']

    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_and_format, batched=True)
    test_dataset = test_dataset.map(tokenize_and_format, batched=True)

    # Save the datasets
    train_dataset.save_to_disk('data/processed/train_dataset_tokenized')
    test_dataset.save_to_disk('data/processed/test_dataset_tokenized')

if __name__ == '__main__':
    # Get the data and process it
    make_dataset()