model=/new_data/yanghq/models/mistralai/Mistral-7B-v0.3
# model=/new_data/yanghq/models/meta-llama/Llama-2-7b-hf

# gpuid=$1
# k_bits=$2
# v_bits=$3
# group_size=$4
# residual_length=$5
# tasks=$6
# model=$7

gpuid=0

anno=16_16_16_16_gsm_kivi_bl_0_0
bit=16

# anno=28_30_30_28_test_ll_1_1
# anno=32_0_32_32_asym_0_0
# anno=32_32_32_32_kivi_0_0
anno=32_32_32_32_gsm_kivi_0_0
# anno=32_0_32_32_gsm_asym_0_0

bit=2

# We report TASK in {coqa, truthfulqa_gen, gsm8k} in our paper.
tasks=coqa
tasks=gsm8k
# tasks=truthfulqa_gen

NUMEXPR_MAX_THREADS='128' CUDA_VISIBLE_DEVICES=$gpuid python run_lm_eval_harness.py --model_name_or_path $model \
    --tasks $tasks \
    --cache_dir ./cached_models \
    --k_bits $bit \
    --v_bits $bit \
    --group_size 32 \
    --residual_length 128 \
    --annotation $anno \

