#!/bin/bash

gpus=(0 1 2 3)
anno_list=(2_32_32_24_32_10 2_32_32_23_32_10 2_32_32_22_32_10 2_32_32_21_32_10 2_32_32_20_32_10 2_31_32_24_32_10 2_31_32_23_32_10 2_31_32_22_32_10 2_31_32_21_32_10 2_31_32_20_32_10 2_30_32_24_32_10 2_30_32_23_32_10 2_30_32_22_32_10 2_30_32_21_32_10 2_30_32_20_32_10 2_29_32_24_32_10 2_29_32_23_32_10 2_29_32_22_32_10 2_29_32_21_32_10 2_29_32_20_32_10 )
anno_list=(2_32_32_16_32_10 2_32_32_14_32_10 2_32_32_12_32_10 2_32_32_10_32_10 2_32_32_8_32_10 2_30_32_16_32_10 2_30_32_14_32_10 2_30_32_12_32_10 2_30_32_10_32_10 2_30_32_8_32_10 2_28_32_16_32_10 2_28_32_14_32_10 2_28_32_12_32_10 2_28_32_10_32_10 2_28_32_8_32_10 2_26_32_16_32_10 2_26_32_14_32_10 2_26_32_12_32_10 2_26_32_10_32_10 2_26_32_8_32_10 )

num_gpus=${#gpus[@]}
num_annos=${#anno_list[@]}
tasks_per_gpu=$(( (num_annos + num_gpus - 1) / num_gpus ))

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
