#!/bin/bash

gpus=(0 7 9 )


anno_list=(0 1 2 3 4 5 6 )
anno_list=(32_32_32_32_q_0 32_32_32_32_q_1 32_32_32_32_q_2 32_32_32_32_q_3 32_32_32_32_q_4 32_32_32_32_q_5 32_32_32_32_q_6 32_32_32_32_q_7 32_32_32_32_q_8 32_32_32_32_q_9 32_32_32_32_q_10 32_32_32_32_q_11 )
anno_list=(24_16_32_32_q_0 24_16_32_32_q_1 24_16_32_32_q_2 24_16_32_32_q_3 24_16_32_32_q_4 32_0_32_32_q_0 32_0_32_32_q_1 32_0_32_32_q_2 32_0_32_32_q_3 32_0_32_32_q_4 )


num_gpus=${#gpus[@]}
num_annos=${#anno_list[@]}

gpu_tasks=()
for ((i = 0; i < num_gpus; i++)); do
    gpu_tasks[i]=0
done

for ((i = 0; i < num_annos; i++)); do
    gpu_index=$((i % num_gpus))
    gpu_tasks[gpu_index]=$((gpu_tasks[gpu_index] + 1))
done

cleanup() {
    echo "收到终止信号，正在停止所有子进程..."
    pkill -P $$  # 终止当前脚本派生的所有子进程
    wait
    echo "所有子进程已停止"
    exit 1
}

trap cleanup SIGINT

start=0
for ((i = 0; i < num_gpus; i++)); do
    gpu=${gpus[$i]}
    tasks_count=${gpu_tasks[i]}
    end=$((start + tasks_count))
    sub_list=("${anno_list[@]:$start:$tasks_count}")
    start=$end

    sub_list_str=$(IFS=","; echo "${sub_list[*]}")
    
    if [ $tasks_count -gt 0 ]; then
        echo "分配任务给 GPU $gpu: ${sub_list[*]}"
        bash ./scripts/run_list_distributed.sh $gpu "$sub_list_str" &
    else
        echo "GPU $gpu 没有分配任务"
    fi
done

wait
echo "Done!"
