model=/new_data/yanghq/models/mistralai/Mistral-7B-Instruct-v0.2
model=/new_data/yanghq/models/meta-llama/Llama-2-7b-chat-hf

# gpuid=$1
# k_bits=$2
# v_bits=$3
# group_size=$4
# residual_length=$5
# tasks=$6
# model=$7

gpuid=9

anno=16_16_16_16_bl_0_0
bit=16

# anno=32_0_32_32_asym_1_1
anno=28_30_30_28_test_ll_1_1
anno=32_0_32_32_asym_0_0
anno=32_32_32_32_kivi_0_0

bit=2

tasks=coqa

# model_name="${model#*/}"
# echo "$model_name"

CUDA_VISIBLE_DEVICES=$gpuid python run_lm_eval_harness.py --model_name_or_path $model \
    --tasks $tasks \
    --cache_dir ./cached_models \
    --k_bits $bit \
    --v_bits $bit \
    --group_size $32 \
    --residual_length $128 \
    --annotation $anno \


