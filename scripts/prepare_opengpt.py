"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import sys
from pathlib import Path
import csv

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import requests
import json
from torch.utils.data import random_split
from lit_llama.tokenizer import Tokenizer
from tqdm import tqdm

TOKEN_USER = "<|user|>" # For chat like interactions we want to have a <user> and <ai> token
TOKEN_AI =  "<|ai|>" # See above
TOKEN_EOS = "<|eos|>" # End of stream (one question, or one answer, or one message)
TOKEN_EOD =  "<|eod|>" # End of document, or conversation - in other words the text that comes after this token is not related to the text before it

DATA_FILES = [
              {
                  "url": "https://raw.githubusercontent.com/CogStack/OpenGPT/main/data/example_project_data/prepared_generated_data_for_example_project.csv",
                  "filename": "prepared_generated_data_for_example_project.csv"
              },
            #   {
            #       "url": "https://raw.githubusercontent.com/CogStack/OpenGPT/main/data/nhs_uk_full/prepared_generated_data_for_nhs_uk_qa.csv",
            #       "filename": "prepared_generated_data_for_nhs_uk_qa.csv"
            #   },
            #   {
            #       "url": "https://raw.githubusercontent.com/CogStack/OpenGPT/main/data/nhs_uk_full/prepared_generated_data_for_nhs_uk_conversations.csv",
            #       "filename": "prepared_generated_data_for_nhs_uk_conversations.csv"
            #   },
            #   {
            #       "url": "https://raw.githubusercontent.com/CogStack/OpenGPT/main/data/medical_tasks_gpt4/prepared_generated_data_for_medical_tasks.csv",
            #       "filename": "prepared_generated_data_for_medical_tasks.csv"
            #   }
              ]
IGNORE_INDEX = -100


def prepare(
    destination_path: Path = Path("data/opengpt"), 
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    test_split_size: int = 20,
    max_seq_length: int = 256,
    seed: int = 42,
    mask_inputs: bool = False,  # as in alpaca-lora
) -> None:
    """Prepare the OpenGPT datasets (healthcare-related) for instruction tuning.
    
    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    
    # TODO: If we don't have the Meta weights, where do we get the tokenizer from?
    tokenizer = Tokenizer(tokenizer_path)
    
    data = []
    destination_path.mkdir(parents=True, exist_ok=True)    
    for data_file in DATA_FILES:
        file_path = destination_path / data_file['filename']
        download(data_file['url'], file_path)         
        
        with open(file_path, "r") as file:
            data.extend([item for item in csv.DictReader(file, delimiter=",")])            

    # Partition the dataset into train and test
    train_split_size = len(data) - test_split_size
    train_set, test_set = random_split(
        data, 
        lengths=(train_split_size, test_split_size),
        generator=torch.Generator().manual_seed(seed),
    )
    train_set, test_set = list(train_set), list(test_set)

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(train_set)]
    torch.save(train_set, file_path.parent / "train.pt")

    print("Processing test split ...")
    test_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(test_set)]
    torch.save(test_set, file_path.parent / "test.pt")


def download(url: str, file_path: Path):
    """Downloads the raw json data file and saves it in the given destination."""
    if file_path.exists():
        return
    with open(file_path, "w") as f:
        f.write(requests.get(url).text)


def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool = True):
    """Processes a single sample.
    
    Each sample in the dataset consists of a <|user|> and an <|ai|> output delimited with <|eos|> and <|eod|> special tokens:
    - text: 
        "<|user|> What is high blood pressure? <|eos|> <|ai|> High blood pressure is a condition where the force at which 
        your heart pumps blood around your body is high. It is recorded with 2 numbers, the systolic pressure and the diastolic pressure, 
        both measured in millimetres of mercury (mmHg).
        References:
        - https://www.nhs.uk/conditions/Blood-pressure-(high)/Pages/Introduction.aspx <|eos|> <|eod|>"
    - raw_data_id: The id of where the instruction originally came from

    This function processes this data to produce an input prompt/target example for
    supervised training. The input text is formed as a single message including all of the interaction between
    the user and the ai.
    The label/target is the same message but can optionally have the user "instruction", i.e., "<|user|> ... <|eos|>"
    masked out (mask_inputs=True).

    Finally, both the input prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenize(tokenizer, full_prompt, max_length=max_length, eos=False)
    encoded_full_prompt_and_response = tokenize(tokenizer, full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[:len(encoded_full_prompt)] = IGNORE_INDEX

    return {**example, "input_ids": encoded_full_prompt_and_response, "input_ids_no_response": encoded_full_prompt, "labels": labels}


def tokenize(tokenizer: Tokenizer, string: str, max_length: int, eos=True) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
