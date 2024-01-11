import pandas as pd 
from sklearn.model_selection import train_test_split


def make_dataset():
    """
    Get the data and process it
    """
    generated_essays = pd.DataFrame()
    # concat all generated essays
    for i in range(1,6):
        generated_essays = pd.concat([generated_essays, pd.read_csv("data/raw/generated_data/AI_Generated_df{}.csv".format(i))])
    generated_essays = generated_essays.rename(columns={"generated_text":"text"})

    # get the original essays
    original_essays = pd.read_csv("data/raw/train_essays.csv")
    original_essays.drop(columns=["id"], inplace=True)

    # concat the two dataframes
    df = pd.concat([original_essays, generated_essays])
    df.reset_index(inplace=True, drop=True)
    # create key from index
    df["key"] = df.index

    # split the data into train and test with sklearn
    X_train, X_test = train_test_split(df[['text','generated']], test_size=0.2, random_state=42)

    # save the data
    X_train.to_csv("data/processed/train.csv", index=False)
    X_test.to_csv("data/processed/test.csv", index=False)

    return X_train, X_test



if __name__ == '__main__':
    # Get the data and process it
    make_dataset()