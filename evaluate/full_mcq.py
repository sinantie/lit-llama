from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from lightning.fabric.strategies import FSDPStrategy
from typing import Optional, Literal, List, Dict
from datasets import load_dataset
from functools import partial
from pathlib import Path
import lightning as L
from tqdm import tqdm
import torch
import warnings
import time
import json
import sys
import os

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_llama import LLaMA, Tokenizer
from lit_llama.model import Block
from lit_llama.utils import EmptyInitOnDevice, chunked_cross_entropy, quantization, lazy_load

def write_results(checkpoint_path: Path,
                  data_file: Path,
                  out_file: Path,
                  acc: float):

    eval_data = {
            "checkpoint": checkpoint_path.name,
            f"{data_file.name}": acc
        }

    with open(out_file, 'a+') as file:

        updated_lines = []
        file.seek(0)

        found = False
        for line in file:
            data = json.loads(line.strip())

            if data.get('checkpoint') == checkpoint_path.name:
                found = True
                data[f"{data_file.name}"] = acc

            updated_lines.append(json.dumps(data))

        file.seek(0)
        file.truncate(0)

        for line in updated_lines:
            file.write(line + os.linesep)

        if not found:
            file.write(json.dumps(eval_data) + os.linesep)


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

def pad_right(x, pad_id, max_len):
    # pad right based on the longest sequence
    n = max_len - len(x)
    return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

def load_mc_data(data_file):
    dataset = load_dataset('json', data_files=str(data_file), split='train',
                           streaming=False)
    samples = list(
        dataset.map(lambda examples: {
            'query': examples['query'],
            'choices': examples['choices'],
            'gold': examples['gold']
        }))
    return samples


def prepare_sample(example: Dict,
                   tokenizer: Tokenizer):


    continuations=[]

    # idx_no_response = tokenizer.encode(f"{example['query']}\nAnswer:", max_length=2048)
    # idx_no_response = tokenizer.encode(full_prompt, max_length=2048)
    idx_no_response = tokenizer.encode(f"{example['query']}Answer:", max_length=2048)

    for choice in example['choices']:
        labels = tokenizer.encode(f"{example['query']}Answer: {choice}", max_length=2048, eos=True)
        # labels = tokenizer.encode(f"{example['query']} {choice}", max_length=2048, eos=True)
        # labels = tokenizer.encode(f"{full_prompt}Answer: {choice}", max_length=2048, eos=True)
        # labels[: len(idx_no_response)] = -1
        continuations.append({"input_ids": idx_no_response, "labels": labels})

    return continuations


def mc_accuracy(fabric: L.Fabric,
                samples: List[Dict],
                model: LLaMA,
                tokenizer: Tokenizer):

    correct = 0

    fabric.print(f"\n{samples[0]['query']}Answer:")
    fabric.print(f"\n----------\n\n{samples[0]['query']}Answer: {samples[0]['choices'][0]}")
    for s in tqdm(samples):

        conts = prepare_sample(s, tokenizer)

        x = torch.stack([pad_right(c["input_ids"].type(torch.int64), pad_id=0, max_len=c['labels'].size(0)) for c in conts])
        y = torch.stack([pad_right(c["labels"].type(torch.int64), pad_id=-1, max_len=c['labels'].size(0)) for c in conts])

        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))

        with torch.no_grad():

            # Forward for perplexity
            logits = model(x)
            losses = []
            for i, l in enumerate(logits):
                losses.append(chunked_cross_entropy(l[..., :-1, :], y[i][..., 1:], chunk_size=0))
            perplexities = torch.exp(torch.stack(losses))
            if perplexities.argmin().item() == s['gold']:
                correct+=1

    return correct / len(samples)

def prepare_cont_tokens_sample(example: Dict,
                               tokenizer: Tokenizer):

    continuations = []

    for choice in example['choices']:
        choice_idxs = tokenizer.encode(f' {choice}')
        idxs = tokenizer.encode(f"{example['query']}Answer: {choice}", max_length=2048)
        continuations.append({"input_ids": idxs, "labels": idxs, "choice_ids": choice_idxs})

    return continuations

def mc_cont_tokens_accuracy(fabric: L.Fabric,
                            samples: List[Dict],
                            model: LLaMA,
                            tokenizer: Tokenizer):

    correct = 0

    fabric.print(f"\n{samples[0]['query']}Answer:")
    fabric.print(f"\n----------\n\n{samples[0]['query']}Answer: {samples[0]['choices'][0]}")
    for s in tqdm(samples):

        conts = prepare_cont_tokens_sample(s, tokenizer)
        xpads = []
        ypads = []
        cspans = []

        for c in conts:

            xpads.append(pad_right(c["input_ids"].type(torch.int64), pad_id=0, max_len=c['labels'].size(0)))
            ypads.append(pad_right(c["labels"].type(torch.int64), pad_id=-1, max_len=c['labels'].size(0)))
            cspans.append(torch.tensor(range(len(c['input_ids']) - len(c['choice_ids']), len(c['input_ids']))))

        x = torch.stack(xpads)
        y = torch.stack(ypads)

        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))

        with torch.no_grad():

            # Forward for perplexity
            logits = model(x)
            losses = []
            for i, (cont_idx, l) in enumerate(zip(cspans, logits)):
                cont_idx = fabric.to_device((cont_idx.pin_memory()))
                cont_tok_logits = l[..., :-1, :].index_select(dim=0, index=cont_idx - 1)
                cont_tok_targ = y[i][..., 1:].index_select(dim=0, index=cont_idx - 1)
                losses.append(chunked_cross_entropy(cont_tok_logits, cont_tok_targ, chunk_size=0))
            perplexities = torch.exp(torch.stack(losses))
            if perplexities.argmin().item() == s['gold']:
                correct+=1

    return correct / len(samples)


def main(
    datasets: str = "wikitext,ptb,c4",
    *,
    # compilation fails as it does not support torch.complex64 for RoPE
    # compile: bool = False,
    data_file: Path = Path(f"checkpoints/stabilityai/stablelm-base-alpha-3b"),
    accelerator: str = "auto",
    strategy: str = "auto",
    devices: List = [0],
    precision: str = "bf16-true",
    checkpoint_path: Optional[Path] = None,
    out_file: Path = None,
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    model_size: str = "7B",
    quantize: Optional[str] = None,    
) -> None:
    """Generates text samples based on a pre-trained LLaMA model and tokenizer.

    Args:
        datasets: The datasets to use as a comma separated string
        # compile: Whether to compile the model.
        accelerator: The hardware to run on. Possible choices are:
            ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
        checkpoint_path: The checkpoint path to load.
        tokenizer_path: The tokenizer path to load.
        dtype: The tensor dtype for choosing the floating-point precision 
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
    """
    if not checkpoint_path:
        checkpoint_path = Path(f"checkpoints/lit-llama/{model_size}/lit-llama.pth")
    assert checkpoint_path.is_file()
    assert tokenizer_path.is_file()

    if strategy == "fsdp":
        auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, cpu_offload=False)
    fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy)
    fabric.seed_everything(1)
    fabric.launch()
    
    # with EmptyInitOnDevice(
    #     device=fabric.device, dtype=dtype, quantization_mode=quantize
    # ):
    #     print("Loading model ...", file=sys.stderr)
    #     t0 = time.time()
    #     model = LLaMA.from_name(model_size)
    #     checkpoint = torch.load(checkpoint_path)
    #     model.load_state_dict(checkpoint)
    #     print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    fabric.print(f"Loading model ...", file=sys.stderr)
    t0 = time.time()
    with fabric.init_module(empty_init=True), quantization(quantize):
        model = LLaMA.from_name(model_size)
    fabric.print(f"Time to instantiate model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    t0 = time.time()

    with lazy_load(checkpoint_path) as checkpoint:
        model.load_state_dict(checkpoint.get("model", checkpoint), strict=quantize is None)
        # model.load_state_dict(checkpoint, strict=False)
    fabric.print(f"Time to load the model weights: {time.time() - t0:.02f} seconds.", file=sys.stderr)
    
    model.eval()

    # if compile:
    #     model = torch.compile(model)

    total_toks = 0
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(tokenizer_path)

    samples = load_mc_data(data_file)

    acc = mc_accuracy(fabric, samples, model, tokenizer)
    # acc = mc_cont_tokens_accuracy(fabric, samples, model, tokenizer)

    fabric.print(f"\nMultiple Choice Accuracy: {acc}\nDataset: {data_file}")
    write_results(checkpoint_path, data_file, out_file, acc)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet",
    )
    CLI(main)
