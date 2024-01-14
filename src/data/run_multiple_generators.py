import argparse
import pandas as pd
import json
from openai import OpenAI
from prompt_generator_data import prompt_data


def main(number_of_prompts, starting_position, csv_file_name_path):
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
        return

    prompt_data(
        api_key=api_key,
        df_prompt_meta=df_prompt_meta,
        df_train_data=df_train_data,
        number_of_prompts=number_of_prompts,
        starting_position=starting_position,
        csv_file_name=csv_file_name_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training prompts for OpenAI.")
    parser.add_argument("--num_prompts", type=int, default=1375, help="Number of prompts to generate")
    parser.add_argument("--start_pos", type=int, default=1000, help="Starting position for prompt generation")
    parser.add_argument(
        "--csv_file_name_path",
        type=str,
        default="data/raw/generated_data/AI_Generated_df1.csv",
        help="Name of the csv file to save the generated prompts to",
    )

    args = parser.parse_args()
    main(args.num_prompts, args.start_pos, args.csv_file_name_path)
