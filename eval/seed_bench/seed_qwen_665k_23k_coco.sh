# The number of available GPUs 
export NPROC_PER_NODE=8

JOB_NAME=qwen_665k_23k_lora_coco
# Produce the Qwen-VL-Chat results of image understanding
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-12345} \
    eval.py \
    --checkpoint /mnt/petrelfs/lijingsong/MLLM/ShareGPT4V/Experiments/Qwen-VL/qwen-665k-23k-lora-coco \
    --dataset image_input.jsonl \
    --batch-size 4 \
    --num-workers 2 \
    --job-name $JOB_NAME
# Collect the result files
cat result_${JOB_NAME}_?.jsonl >$JOB_NAME.jsonl
rm result_${JOB_NAME}_?.jsonl

# # Produce the results of video understanding
# python -m torch.distributed.launch --use-env \
#     --nproc_per_node ${NPROC_PER_NODE:-8} \
#     --nnodes ${WORLD_SIZE:-1} \
#     --node_rank ${RANK:-0} \
#     --master_addr ${MASTER_ADDR:-127.0.0.1} \
#     --master_port ${MASTER_PORT:-12345} \
#     eval.py \
#     --checkpoint Qwen/Qwen-VL-Chat \
#     --dataset video_input_4.jsonl \
#     --batch-size 2 \
#     --num-workers 1
# # Collect the result files
# cat result_?.jsonl >results_chat_vid.jsonl
# rm result_?.jsonl

# The file `results_chat.jsonl` can be submitted to the leaderboard
# cat results_chat_img.jsonl results_chat_vid.jsonl >results_chat.jsonl