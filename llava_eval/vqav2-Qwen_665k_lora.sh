#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="qwen_665k_lora"
SPLIT="llava_vqav2_mscoco_test-dev2015"

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#         --model-path /mnt/petrelfs/lijingsong/MLLM/ShareGPT4V/Experiments/Qwen-VL/qwen-665k-lora \
#         --question-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/vqav2/$SPLIT.jsonl \
#         --image-folder /mnt/petrelfs/lijingsong/MLLM/Benchmark/vqav2/test2015 \
#         --answers-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode vicuna_v1 &
# done

# wait

# output_file=/mnt/petrelfs/lijingsong/MLLM/Benchmark/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat /mnt/petrelfs/lijingsong/MLLM/Benchmark/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

