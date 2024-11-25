#!/bin/bash

gpus=(0 1 2 3 4 )
anno_list=(16_32_32_32_q_0 16_32_32_32_q_1 )
anno_list=(0 1 2 3 4 5 )

num_gpus=${#gpus[@]}
num_annos=${#anno_list[@]}
tasks_per_gpu=$(( (num_annos + num_gpus - 1) / num_gpus ))

cleanup() {
    echo "收到终止信号，正在停止所有子进程..."
    pkill -P $$  # 终止当前脚本派生的所有子进程
    wait
    echo "所有子进程已停止"
    exit 1
}

trap cleanup SIGINT

for ((i=0; i<$num_gpus; i++)); do
    gpu=${gpus[$i]}
    start=$((i * tasks_per_gpu))
    end=$((start + tasks_per_gpu))

    if [ $end -gt $num_annos ]; then
        end=$num_annos
    fi

    sub_list=("${anno_list[@]:$start:$((end - start))}")

    sub_list_str=$(IFS=","; echo "${sub_list[*]}")

    echo "分配任务给 GPU $gpu: ${sub_list[*]}"
    # CUDA_VISIBLE_DEVICES=$gpu bash ./scripts/run_list.sh $gpu "$sub_list_str" &
    bash ./scripts/run_list_distributed.sh $gpu "$sub_list_str" &
done

wait
echo "Done!"
