#!/bin/bash

JOB_NAME=qwen

python -m llava.eval.model_vqa_science \
    --model-path /mnt/petrelfs/lijingsong/MLLM/Models/Qwen-VL-Chat \
    --question-file /mnt/petrelfs/lijingsong/MLLM/ShareGPT4V/LLaVA/playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /mnt/petrelfs/lijingsong/MLLM/ShareGPT4V/LLaVA/playground/data/eval/scienceqa/images/test \
    --answers-file /mnt/petrelfs/lijingsong/MLLM/ShareGPT4V/LLaVA/playground/data/eval/scienceqa/answers/${JOB_NAME}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir /mnt/petrelfs/lijingsong/MLLM/ShareGPT4V/LLaVA/playground/data/eval/scienceqa \
    --result-file /mnt/petrelfs/lijingsong/MLLM/ShareGPT4V/LLaVA/playground/data/eval/scienceqa/answers/${JOB_NAME}.jsonl \
    --output-file /mnt/petrelfs/lijingsong/MLLM/ShareGPT4V/LLaVA/playground/data/eval/scienceqa/answers/${JOB_NAME}_output.jsonl \
    --output-result /mnt/petrelfs/lijingsong/MLLM/ShareGPT4V/LLaVA/playground/data/eval/scienceqa/answers/${JOB_NAME}_result.json