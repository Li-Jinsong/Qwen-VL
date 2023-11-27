#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT='qwen_158k_lora'

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path /mnt/petrelfs/lijingsong/MLLM/ShareGPT4V/Experiments/Qwen-VL/qwen-158k-lora \
        --question-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/seed_bench/llava-seed-bench.jsonl \
        --image-folder /mnt/petrelfs/lijingsong/MLLM/Benchmark/seed_bench \
        --answers-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=/mnt/petrelfs/lijingsong/MLLM/Benchmark/seed_bench/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /mnt/petrelfs/lijingsong/MLLM/Benchmark/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/convert_seed_for_submission.py \
    --annotation-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/seed_bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/seed_bench/answers_upload/$CKPT.jsonl \
    -t

