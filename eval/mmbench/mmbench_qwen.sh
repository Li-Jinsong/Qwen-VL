JOB_NAME=qwen_baseline

checkpoint=/mnt/petrelfs/lijingsong/MLLM/Models/Qwen-VL-Chat
ds=mmbench_dev_20230712

# python -m torch.distributed.launch --use-env \
#     --nproc_per_node ${NPROC_PER_NODE:-8} \
#     --nnodes ${WORLD_SIZE:-1} \
#     --node_rank ${RANK:-0} \
#     --master_addr ${MASTER_ADDR:-127.0.0.1} \
#     --master_port ${MASTER_PORT:-12345} \
#     evaluate_multiple_choice_mmbench.py \
#     --checkpoint $checkpoint \
#     --dataset $ds \
#     --job-name $JOB_NAME \
#     --batch-size 2 \
#     --num-workers 2

# without consistency constrain

# python mmbench_evaluation.py \
#     --job-name $JOB_NAME

# with consistency constrain

python mmbench_evaluation_tricky.py \
    --job-name $JOB_NAME

# python mmbench_predict_to_submission.py \
#     --job-name $JOB_NAME