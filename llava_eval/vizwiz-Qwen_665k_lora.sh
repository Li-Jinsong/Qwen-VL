#!/bin/bash

JOB_NAME=qwen_665k_lora

# python -m llava.eval.model_vqa_loader \
#     --model-path /mnt/petrelfs/lijingsong/MLLM/ShareGPT4V/Experiments/Qwen-VL/qwen-665k-lora \
#     --question-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/vizwiz/llava_test.jsonl \
#     --image-folder /mnt/petrelfs/lijingsong/MLLM/Benchmark/vizwiz/test \
#     --answers-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/vizwiz/answers/$JOB_NAME.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/vizwiz/llava_test.jsonl \
    --result-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/vizwiz/answers/$JOB_NAME.jsonl \
    --result-upload-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/vizwiz/answers_upload/$JOB_NAME.json
