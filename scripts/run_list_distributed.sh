#!/bin/bash

gpuid=$1
anno_list=$2
bit=${3:-"2"}

model=/new_data/yanghq/models/mistralai/Mistral-7B-Instruct-v0.2
model=/new_data/yanghq/models/meta-llama/Llama-2-7b-chat-hf

# anno_list=(2_30_32_16_32_10 2_30_32_16_32_11 2_30_32_16_32_12 2_30_32_16_32_13 2_30_32_16_32_14 2_30_32_16_32_15 2_32_32_14_32_10 2_32_32_14_32_11 2_32_32_14_32_12 2_32_32_14_32_13 2_32_32_14_32_14 2_32_32_14_32_15 )


IFS=',' read -r -a anno_array <<< "$anno_list"

cleanup() {
    echo "正在终止 GPU #${gpuid} 的所有任务..."
    pkill -P $$  # 终止当前脚本的所有子进程
    exit 1
}

trap cleanup SIGINT

sleep 1
echo "num_task = ${#anno_array[*]} in gpu#${gpuid}"
sleep 1


for anno in ${anno_array[@]}
do
    # echo "running ${anno} in ${gpuid}"
    sleep 1


    CUDA_VISIBLE_DEVICES=$gpuid python q_pred_long_bench.py --model_name_or_path $model \
        --cache_dir ./cached_models \
        --k_bits $bit \
        --v_bits $bit \
        --k_quant_dim 'channel' \
        --v_quant_dim 'token' \
        --group_size 32 \
        --residual_length 128 \
        --annotation $anno

    python3 eval.py --model_name_or_path $model \
        --cache_dir ./cached_models \
        --k_bits $bit \
        --v_bits $bit \
        --k_quant_dim 'channel' \
        --v_quant_dim 'token' \
        --group_size 32 \
        --residual_length 128 \
        --annotation $anno

    python3 send.py --model_name_or_path $model \
        --cache_dir ./cached_models \
        --k_bits $bit \
        --v_bits $bit \
        --k_quant_dim 'channel' \
        --v_quant_dim 'token' \
        --group_size 32 \
        --residual_length 128 \
        --annotation $anno


done

wait