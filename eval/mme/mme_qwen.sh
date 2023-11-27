JOB_NAME=qwen_baseline

cd /mnt/petrelfs/lijingsong/MLLM/Benchmark/MME/eval_tool/

python qwen_eval.py \
    --checkpoint /mnt/petrelfs/lijingsong/MLLM/Models/Qwen-VL-Chat \
    --job-name ${JOB_NAME}
python calculation.py --results_dir ${JOB_NAME}