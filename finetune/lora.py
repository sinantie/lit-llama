"""
Instruction-tuning with LoRA on the Alpaca dataset.

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import sys
from pathlib import Path
import os
import time

import lightning as L
import numpy as np
import torch
import wandb

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama.lora import mark_only_lora_as_trainable, lora, lora_state_dict
from lit_llama.model import LLaMA, LLaMAConfig
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt
    
def main(
    instruction_tuning: bool = True, # set to true for instruction-style examples; only used for validation purposes
    # logging parameters
    eval_interval: int = 100,
    save_interval: int = 100,
    eval_iters: int = 100,
    log_interval: int = 100,
    # Hyperparameters
    learning_rate: float = 3e-4,
    batch_size: int = 128,
    micro_batch_size: int = 4,    
    weight_decay: float = 0.0,
    max_seq_length: int = 256,  # see scripts/prepare_alpaca.py
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    warmup_iters: int = 100,
    # project paths etc
    project_name: str = "open-llama-alpaca-lora-7b",
    train_dataset_dir: str = "data/alpaca/train.pt", 
    val_dataset_dir: str = "data/alpaca/test.pt", 
    pretrained_path: str = "checkpoints/lit-llama/7B/lit-llama.pth",
    tokenizer_path: str = "checkpoints/lit-llama/tokenizer.model",
    out_dir: str = "out/lora/alpaca",
    example_instruction: str = "Recommend a movie for me to watch during the weekend and explain the reason.",
):
    gradient_accumulation_iters = batch_size // micro_batch_size    
    assert gradient_accumulation_iters > 0
    max_iters = 50000 * 3 // micro_batch_size
    
    logging_params = {
        "eval_interval": eval_interval,
        "save_interval": save_interval,
        "eval_iters": eval_iters,
        "log_interval": log_interval,
    }
    
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "architecture": "lora",
        "dataset": train_dataset_dir,
        "batch_size": batch_size,
        "max_iters": max_iters,
        "max_seq_length": max_seq_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "warmup_iters": warmup_iters
        }
    )
    fabric = L.Fabric(accelerator="cuda", devices=1, precision="bf16-true")
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets(train_dataset_dir=train_dataset_dir, val_dataset_dir=val_dataset_dir)

    config = LLaMAConfig.from_name("7B")
    config.block_size = max_seq_length

    checkpoint = torch.load(pretrained_path)

    with fabric.init_module(), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
        model = LLaMA(config)
        # strict=False because missing keys due to LoRA weights not contained in checkpoint state
        model.load_state_dict(checkpoint, strict=False)
    
    mark_only_lora_as_trainable(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model, optimizer = fabric.setup(model, optimizer)
    # needed only to predict a validation sample example
    tokenizer = Tokenizer(tokenizer_path)
    
    train(fabric, model, optimizer, train_data, val_data, tokenizer, out_dir, 
          max_iters, warmup_iters, learning_rate, gradient_accumulation_iters, example_instruction, instruction_tuning, 
          max_seq_length, micro_batch_size, logging_params)

    # Save the final LoRA checkpoint at the end of training
    checkpoint = lora_state_dict(model)
    fabric.save(os.path.join(out_dir, "lit-llama-lora-finetuned.pth"), checkpoint)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    tokenizer: Tokenizer,
    out_dir: str,
    max_iters: int,
    warmup_iters: int,
    learning_rate: float,
    gradient_accumulation_iters: int,
    example_instruction: str,
    instruction_tuning: bool, 
    max_seq_length: int, 
    micro_batch_size: int,
    logging_params: dict
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0

    for iter_num in range(max_iters):

        if step_count <= warmup_iters:
            # linear warmup
            lr = learning_rate * step_count / warmup_iters
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        input_ids, targets = get_batch(fabric, train_data, micro_batch_size)
        with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_iters != 0)):
            logits = model(input_ids)
            loss = loss_fn(logits, targets)
            fabric.backward(loss / gradient_accumulation_iters)

        if (iter_num + 1) % gradient_accumulation_iters == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
            wandb.log({"train": {"loss": loss.item()}}, step=step_count)

            if step_count % logging_params["eval_interval"] == 0:
                val_loss = validate(fabric, model, val_data, tokenizer, example_instruction, instruction_tuning, max_seq_length, micro_batch_size, logging_params)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                wandb.log({"val": {"loss": val_loss}}, step=step_count)
                fabric.barrier()

            if step_count % logging_params["save_interval"] == 0:
                print(f"Saving LoRA weights to {out_dir}")
                # We are only saving the LoRA weights
                # TODO: Provide a function/script to merge the LoRA weights with pretrained weights
                checkpoint = lora_state_dict(model)
                fabric.save(os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth"), checkpoint)

        dt = time.time() - t0
        if iter_num % logging_params["log_interval"] == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")        


def generate_response(model: torch.nn.Module, 
                        instruction: str,
                        tokenizer: Tokenizer,
                        instruction_tuning: bool,
                        max_seq_length: int) -> str:    
    sample = {"instruction": instruction, "input": ""}
    prompt = instruction
    if instruction_tuning: # for alpaca-style instructions only
        prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=max_seq_length,
        max_new_tokens=100,
    )
    output = tokenizer.decode(output)
    return output # output.split("### Response:")[1].strip()


@torch.no_grad()
def validate(fabric: L.Fabric, 
             model: torch.nn.Module, 
             val_data: np.ndarray, 
             tokenizer: Tokenizer, 
             example_instruction: str, 
             instruction_tuning: bool, 
             max_seq_length: int,
             micro_batch_size: int,
             logging_params: dict) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(logging_params["eval_iters"])
    for k in range(logging_params["eval_iters"]):
        input_ids, targets = get_batch(fabric, val_data, micro_batch_size)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    out = losses.mean()

    # produce an example:
    instruction = example_instruction
    
    output = generate_response(model, instruction, tokenizer, instruction_tuning, max_seq_length)
    fabric.print(instruction)
    fabric.print(output)

    model.train()
    return out.item()

def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss
    

def get_batch(fabric: L.Fabric, data: list, micro_batch_size: int):
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(train_dataset_dir: str, val_dataset_dir: str):
    train_data = torch.load(train_dataset_dir)
    val_data = torch.load(val_dataset_dir)
    return train_data, val_data


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    
    from jsonargparse.cli import CLI

    CLI(main)
