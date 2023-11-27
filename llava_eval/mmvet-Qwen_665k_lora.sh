#!/bin/bash

JOB_NAME=qwen-665k-lora

python -m llava.eval.model_vqa \
    --model-path /mnt/petrelfs/lijingsong/MLLM/ShareGPT4V/Experiments/Qwen-VL/$JOB_NAME \
    --question-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/mm-vet/llava-mm-vet.jsonl \
    --image-folder /mnt/petrelfs/lijingsong/MLLM/Benchmark/mm-vet/images \
    --answers-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/mm-vet/answers/$JOB_NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /mnt/petrelfs/lijingsong/MLLM/Benchmark/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src /mnt/petrelfs/lijingsong/MLLM/Benchmark/mm-vet/answers/$JOB_NAME.jsonl \
    --dst /mnt/petrelfs/lijingsong/MLLM/Benchmark/mm-vet/results/$JOB_NAME.json

