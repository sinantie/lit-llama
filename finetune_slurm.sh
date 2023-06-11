#!/bin/bash -l
#flight env activate conda@lit-llama

srun --nodes=1 --mem=32768 --time=04:00:00 --partition=gpu --gres=gpu:1 --output srun.out python finetune/lora.py --config config/finetune_lora_opengpt.yaml 