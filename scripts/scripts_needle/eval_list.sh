# This script is adapted from
# https://github.com/FranxYao/Long-Context-Data-Engineering.git

mkdir -p ./results_needle/logs/
mkdir -p ./results_needle/img/
MAX_CAPACITY_PROMPT=96  # [64, 96, 128, 256, 512, 1024, 2048, ...]
attn_implementation="flash_attention_2" # Support "flash_attention_2", "sdpa", "None".
MODEL=/new_data/yanghq/models/mistralai/Mistral-7B-Instruct-v0.2

# ['full', 'pyramidkv', 'snapkv', 'streamingllm', 'h2o', 'cam']
METHOD='full'
METHOD='ours'

# TAG=test


STEP=2550
MIN_LEN=1000
MAX_LEN=26501

# anno_list=(32_0_32_16_mi0_0_0 32_0_32_16_mi0_1_0 32_0_32_16_mi0_2_0 30_2_32_16_mi1_0_0 30_2_32_16_mi1_1_0 30_2_32_16_mi1_2_0 )
# gpuid=0

# anno_list=(32_0_32_16_mi0_0_1 32_0_32_16_mi0_1_1 32_0_32_16_mi0_2_1 30_2_32_16_mi1_0_1 30_2_32_16_mi1_1_1 30_2_32_16_mi1_2_1 )
# gpuid=1

# anno_list=(32_0_32_16_mi0_0_2 32_0_32_16_mi0_1_2 32_0_32_16_mi0_2_2 30_2_32_16_mi1_0_2 30_2_32_16_mi1_1_2 30_2_32_16_mi1_2_2 )
# gpuid=3

anno_list=(32_0_32_32_asym_0_0 32_32_32_32_kivi_0_0 )
gpuid=9


echo "numbers of array = ${#anno_list[*]}"

for anno in ${anno_list[@]}
do
    echo $anno

    (
    CUDA_VISIBLE_DEVICES=$gpuid python -u run_needle_in_haystack.py --s_len $MIN_LEN --e_len $MAX_LEN \
        --model_provider Mistral \
        --model_name $MODEL \
        --step $STEP \
        --method $METHOD \
        --max_capacity_prompt $MAX_CAPACITY_PROMPT \
        --model_version Mistral2_${METHOD}_${anno} \
        --annotation $anno
    ) 2>&1  | tee ./results_needle/logs/Mistral2_${METHOD}_${anno}.log

    python ./scripts/scripts_needle/visualize.py --result_path ./results_needle/results/Mistral2_${METHOD}_${anno}/

done