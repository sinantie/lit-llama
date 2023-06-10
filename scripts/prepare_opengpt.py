"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import sys
from pathlib import Path
import csv
from typing import List

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import requests
from torch.utils.data import random_split
from lit_llama.tokenizer import Tokenizer
from tqdm import tqdm

TOKEN_USER = "<|user|>" # For chat like interactions we want to have a <user> and <ai> token
TOKEN_AI =  "<|ai|>" # See above
TOKEN_EOS = "<|eos|>" # End of stream (one question, or one answer, or one message)
TOKEN_EOD =  "<|eod|>" # End of document, or conversation - in other words the text that comes after this token is not related to the text before it

DATA_FILES = [
              {
                  "purpose": "debug",
                  "url": "https://raw.githubusercontent.com/CogStack/OpenGPT/main/data/example_project_data/prepared_generated_data_for_example_project.csv",
                  "filename": "prepared_generated_data_for_example_project.csv"
              },
              {
                  "purpose": "single_shot-qa",
                  "url": "https://raw.githubusercontent.com/CogStack/OpenGPT/main/data/nhs_uk_full/prepared_generated_data_for_nhs_uk_qa.csv",
                  "filename": "prepared_generated_data_for_nhs_uk_qa.csv"
              },
              {
                  "purpose": "conversational-qa",
                  "url": "https://raw.githubusercontent.com/CogStack/OpenGPT/main/data/nhs_uk_full/prepared_generated_data_for_nhs_uk_conversations.csv",
                  "filename": "prepared_generated_data_for_nhs_uk_conversations.csv"
              },
              {
                  "purpose": "single_shot-qa",
                  "url": "https://raw.githubusercontent.com/CogStack/OpenGPT/main/data/medical_tasks_gpt4/prepared_generated_data_for_medical_tasks.csv",
                  "filename": "prepared_generated_data_for_medical_tasks.csv"
              }
              ]
IGNORE_INDEX = -100

dataset_stats = {
        "total_context_tokens": 0,
        "max_context_length": 0,
        "total_prompt_tokens": 0,
        "max_prompt_length": 0,
    }

def prepare(
    destination_path: Path = Path("data/opengpt"), 
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    test_split_size: int = 2000,
    max_seq_length: int = 256,
    seed: int = 42,
    mask_inputs: bool = False,  # as in alpaca-lora
    partitions_to_include: List[str] = ['single_shot-qa'], # options are  ['debug', 'single_shot-qa', 'conversational-qa']
    split_conversations_to_examples: bool = False,
    
) -> None:
    """Prepare the OpenGPT datasets (healthcare-related) for instruction tuning.

    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.

    Args:
        destination_path: Path to download varius corpus files and destination datasets.
        tokenizer_path: Path to the Tokenizer.
        test_split_size:    Number of examples to use for the validation set.
        max_seq_length: Maximum sequence length for the overall prompt (including the response).
        seed:   The default seed used for randomly splitting the datasets.
        mask_inputs:    Whether to ignore (apply mask) the input prompt (i.e., context with the <|user|> query, or conversation turns up to the final (but not including) <|ai|> response).
        partitions_to_include:  Which partitions of the dataset to include. Available options are ['debug', 'single_shot-qa' (DEFAULT), 'conversational-qa']
        split_conversations_to_examples:  Split a multi-turn conversational example into multiple individual ones incrementally.
    """
    
    # TODO: If we don't have the Meta weights, where do we get the tokenizer from?
    tokenizer = Tokenizer(tokenizer_path)
    
    data = []
    destination_path.mkdir(parents=True, exist_ok=True)    
    filtered_data_files = [data_file for data_file in DATA_FILES if data_file['purpose'] in partitions_to_include]
    for data_file in filtered_data_files:
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

    train_length = len(train_set)
    print(f"train has {train_length:,} samples")
    print(f"val has {len(test_set):,} samples")           
    
    print("Processing train split ...")
    train_set = [x for x in filter(None, [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(train_set)])]
    torch.save(train_set, file_path.parent / "train.pt")

    print("Processing test split ...")
    test_set = [x for x in filter(None, [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(test_set)])]
    torch.save(test_set, file_path.parent / "test.pt")
 
    total_processed_data = len(train_set) + len(test_set)
    print(f"Average words per context: {(dataset_stats['total_context_tokens'] / total_processed_data) :,}. Largest context: {dataset_stats['max_context_length']} words.")
    print(f"Average words per prompt: {(dataset_stats['total_prompt_tokens'] / total_processed_data) :,}. Largest prompt: {dataset_stats['max_prompt_length']} words.")    

def download(url: str, file_path: Path):
    """Downloads the raw data file and saves it in the given destination."""
    if file_path.exists():
        return
    with open(file_path, "w") as f:
        f.write(requests.get(url).text)


def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool = True):
    """Processes a single sample.
    
    Each sample in the dataset consists of potentially multiple <|user|> queries and <|ai|> outputs delimited with <|eos|>; 
    the end of an interaction/dialogue is indicated by <|eod|> special tokens:
    - text: 
        "<|user|> What is high blood pressure? <|eos|> <|ai|> High blood pressure is a condition where the force at which 
        your heart pumps blood around your body is high. It is recorded with 2 numbers, the systolic pressure and the diastolic pressure, 
        both measured in millimetres of mercury (mmHg).
        References:
        - https://www.nhs.uk/conditions/Blood-pressure-(high)/Pages/Introduction.aspx <|eos|> <|eod|>"
    - raw_data_id: The id of where the instruction originally came from

    This function processes this data to produce an input context/target example for
    supervised training. The input text is formed as a single message including all of the interaction between
    the user and the ai (context).
    The label/target is the same message but can optionally have the user "instruction", i.e., "<|user|> ... <|eos|>"
    masked out (mask_inputs=True).

    Finally, both the input prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    context, response = generate_prompt(example)
    """An error occurred in processing the example so don't continue"""
    if not context:
        return
    context_and_response = context + response
    
    compute_stats(len(context.split()), len(response.split()))
    
    encoded_context = tokenize(tokenizer, context, max_length=max_length, eos=False)
    encoded_context_and_response = tokenize(tokenizer, context_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_context_and_response.clone()
    if mask_inputs:
        labels[:len(encoded_context)] = IGNORE_INDEX

    return {**example, "input_ids": encoded_context_and_response, "input_ids_no_response": encoded_context, "labels": labels}

def compute_stats(count_context_tokens: int, count_response_tokens: int) -> None:  
    max = dataset_stats['max_context_length']
    dataset_stats['max_context_length'] = count_context_tokens if max < count_context_tokens else max
    dataset_stats['total_context_tokens'] += count_context_tokens    

    count_prompt_tokens = count_context_tokens + count_response_tokens
    max = dataset_stats['max_prompt_length']
    dataset_stats['max_prompt_length'] = count_prompt_tokens if max < count_prompt_tokens else max
    dataset_stats['total_prompt_tokens'] += count_prompt_tokens    

def tokenize(tokenizer: Tokenizer, string: str, max_length: int, eos=True) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)


def generate_prompt(example):
    """Generates a prompt with the context of the dialogue or single user query and the response seperately."""    
    text_split = example['text'].rsplit(TOKEN_AI, 1)
    if len(text_split) > 1:
        return text_split[0], TOKEN_AI + text_split[1]
    else:    
        print(f"Invalid example! id: {example['raw_data_id']}\ntext: {example['text']}")
        return "", ""
    

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
