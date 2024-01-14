import pandas as pd
import random
import logging
from rich.logging import RichHandler
import json
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

rich_handler = RichHandler(markup=True)
rich_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(rich_handler)


# Handlings api errorss
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def prompt_data(
    api_key: str,
    df_prompt_meta: pd.DataFrame,
    df_train_data: pd.DataFrame,
    number_of_prompts: int,
    starting_position: int = 0,
    csv_file_name: str = "data/raw/generated_data/AI_Generated_df.csv",
):
    """
    promt_data generates a prompt for each essay in the training data and saves a DataFrame with the generated prompt and the label

        Parameters:
                api_key (str): Key for OpenAI API
                df_prompt_meta (pd.DataFrame): DataFrame with the prompt meta data
        df_train_data (pd.DataFrame): DataFrame with the training data
        number_of_prompts (int): Number of prompts to generate

        Returns:
                None
    """

    client = OpenAI(api_key=api_key)

    AI_Generated_df = pd.DataFrame({"generated_text": [], "generated": [], "prompt_id": []})

    for i, row in df_train_data.iterrows():
        if row["generated"] == 1:
            continue

        if i >= number_of_prompts:
            break

        logger.info("--------------------------------------------------------------")
        logger.info("Generating prompt for essay " + str(i) + " of " + str(number_of_prompts) + " prompts")

        student_essay = df_train_data["text"][i]

        ["Professionally", "Humanly"]

        full_prompt = "A student wrote an essay:\n" + student_essay + "\n" + "Please continue this essay:"

        logger.debug("Prompt ID: " + str(row["prompt_id"]))

        # Generate completion using OpenAI's API
        completion = completion_with_backoff(
            client=client,
            model="gpt-3.5-turbo",  # Either gpt-4-1106-preview or gpt-3.5-turbo
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.99,
        )

        output = completion.choices[0].message.content

        # Add the generated output to the DataFrame
        new_df = pd.DataFrame({"generated_text": [output], "generated": [1], "prompt_id": [row["prompt_id"]]})
        AI_Generated_df = pd.concat([AI_Generated_df, new_df], ignore_index=True)
        AI_Generated_df.to_csv(csv_file_name, index=False)

        logger.info(f"\n{output}")


if __name__ == "__main__":
    df_prompt_meta = pd.read_csv("data/raw/train_prompts.csv")
    df_train_data = pd.read_csv("data/raw/train_essays.csv")

    try:
        json_file_path = "api_key.json"
        with open(json_file_path, "r") as json_file:
            api_key_json = json.load(json_file)
        api_key = api_key_json["api_key"]
    except FileNotFoundError:
        print(
            "No api_key.json file found. Please create a file with your OpenAI API key and save it as api_key.json in the root folder."
        )

    number_of_prompts = 1375

    client = OpenAI(api_key=api_key)

    prompt_data(
        api_key=api_key,
        df_prompt_meta=df_prompt_meta,
        df_train_data=df_train_data,
        number_of_prompts=number_of_prompts,
        starting_position=1000,
    )
