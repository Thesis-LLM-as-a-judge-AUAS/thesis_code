import os
import json


def build_dataset(data_type, data_path="./data"):
    """
    Load and construct a dataset based on the specified data_type.

    Only supports:
        - "verbosity"
        - "vicuna"
        - "vicuna-mec"
        - "vicuna-gpt4"
        - "vicuna-mec-gpt4"

    Parameters:
    - data_type (str): The type of dataset to load.
    - data_path (str): The base directory where datasets are stored.

    Returns:
    - dataset (list): A list of examples in standardized format.
    """

    # Ensure only supported data types are used
    assert data_type in ["verbosity", "vicuna", "vicuna-mec", "vicuna-gpt4", "vicuna-mec-gpt4"], \
        f"Unsupported data_type: {data_type}"

    # Case 1: Handle all Vicuna-related datasets
    if data_type in ['vicuna', 'vicuna-mec', 'vicuna-gpt4', 'vicuna-mec-gpt4']:
        # Map each data_type to its corresponding JSON file path
        data_type_to_path = {
            'vicuna': 'vicuna/vanilla-vicuna.json',
            'vicuna-mec': 'vicuna/mec-bpc-vicuna.json',
            'vicuna-gpt4': 'vicuna/vanilla-vicuna-gpt4.json',
            'vicuna-mec-gpt4': 'vicuna/mec-bpc-vicuna-gpt4.json',
        }

        # Open the JSON file corresponding to the selected data_type
        with open(os.path.join(data_path, data_type_to_path[data_type]), "r", encoding="utf-8") as fin:
            dataset = json.load(fin)  # Load JSON data as a list of records

        new_dataset = []
        for line in dataset:
            # Construct a standardized example format with question and two answers
            example = {
                'question_body': line['question_text'],
                'answer1_body': line['text_gpt'],  # GPT-generated response
                'answer2_body': line['text_vicuna']  # Vicuna-generated response
            }
            new_dataset.append(example)

        dataset = new_dataset  # Final structured dataset

    # Case 2: Handle Verbosity dataset
    elif data_type == "verbosity":
        # Open the verbosity dataset JSON file
        with open(os.path.join(data_path, "verbosity/judgement_expanded.json"), "r", encoding="utf-8") as fin:
            dataset = json.load(fin)  # Load JSON data

        new_dataset = []
        for line in dataset:
            # Standardize format to match other datasets
            example = {
                'question_body': line['instruction'],  # The task instruction
                'answer1_body': line['answer_1'],  # First model's response
                'answer2_body': line['answer_2']  # Second model's response
            }
            new_dataset.append(example)

        dataset = new_dataset  # Final structured dataset

    # Return the loaded and standardized dataset
    return dataset
