import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama import Tokenizer, LLaMA
from lit_llama.lora import lora
from lit_llama.utils import lazy_load, llama_model_lookup
from scripts.prepare_alpaca import generate_prompt as alpaca_prompt
from scripts.prepare_opengpt import generate_prompt as opengpt_prompt

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05


def main(
    lora_path: Path = Path("out/lora/alpaca/lit-llama-lora-finetuned.pth"),
    pretrained_path: Path = Path("checkpoints/lit-llama/7B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    quantize: Optional[str] = None,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    instruction_tuning: bool = True, # set to true for instruction-style examples; otherwise (default) to chatbot mode.
    special_tokens: dict[str, str] = {"user": "<|user|>", "ai": "<|ai|>", "eos": "<|eos|>", "eod": "<|eod|>"},
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned LoRA model.
    See `finetune_lora.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        lora_path: Path to the checkpoint with trained LoRA weights, which are the output of
            `finetune_lora.py`.
        input: Optional input (Alpaca style).
        pretrained_path: The path to the checkpoint with pretrained LLaMA weights.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
    """
    assert lora_path.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()

    if quantize is not None:
        raise NotImplementedError("Quantization in LoRA is not supported yet")

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()

    with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(lora_path) as lora_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with fabric.init_module(empty_init=True), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
            model = LLaMA.from_name(name)

            # 1. Load the pretrained weights
            model.load_state_dict(pretrained_checkpoint, strict=False)
            # 2. Load the fine-tuned lora weights
            model.load_state_dict(lora_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)

    tokenizer = Tokenizer(tokenizer_path)

    user_message = ""
    while user_message != 'quit':
        user_message = input('User: ')
        output, total_time = generate_response(user_message, tokenizer, model, max_new_tokens, temperature, top_k, 
                                               special_tokens, instruction_tuning, fabric.device.type == "cuda", debug=False)
        print(f"AI [{total_time:.02f} sec]: {output} ")

def generate_response(user_message: str,                 
                tokenizer: Tokenizer, 
                model: LLaMA,
                max_new_tokens: int, 
                temperature: float,
                top_k: int,
                special_tokens: dict[str, str],
                instruction_tuning: bool = False, 
                use_cuda: bool = False,
                debug: bool = False
                ):
    if instruction_tuning:
        if 'Input:' in user_message:
            instruction, user_input = user_message.split('Input:')
            sample = {"instruction": instruction, "input": user_input}
        else:
            sample = {"instruction": instruction}
        prompt = alpaca_prompt(sample)
    else:
        sample = f"{special_tokens['user']} {user_message} {special_tokens['eos']} {special_tokens['ai']}"
        prompt, _ = opengpt_prompt({"text": sample}, special_tokens_input=special_tokens, special_tokens_output=None)    
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
    
    t0 = time.perf_counter()
    output = generate(
        model,
        idx=encoded,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.eos_id
    )
    t = time.perf_counter() - t0
    model.reset_cache()
    output = tokenizer.decode(output)
    if instruction_tuning:
        output = output.split("### Response:")[1].strip()
    else:            
        output = output.rsplit(f"{special_tokens['eos']} {special_tokens['ai']}", 1)[1]        

    if debug:
        print(f"\n\nTime for inference: {t:.02f} sec total, {max_new_tokens / t:.02f} tokens/sec", file=sys.stderr)
        if use_cuda:
            print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)
    return output, t


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)