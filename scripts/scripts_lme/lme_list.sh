model=/new_data/yanghq/models/mistralai/Mistral-7B-v0.3
model=/new_data/yanghq/models/meta-llama/Llama-2-7b-hf

# gpuid=$1
# k_bits=$2
# v_bits=$3
# group_size=$4
# residual_length=$5
# tasks=$6
# model=$7

gpuid=0

# anno=28_30_30_28_test_ll_1_1
# anno=32_0_32_32_asym_0_0
# anno=32_32_32_32_kivi_0_0

bit=2

# We report TASK in {coqa, truthfulqa_gen, gsm8k} in our paper.
tasks=coqa
# tasks=gsm8k
# tasks=truthfulqa_gen


# anno_list=(32_0_32_16_coqa_mi0_0_0 32_0_32_16_coqa_mi0_1_0 32_0_32_16_coqa_mi0_2_0 30_2_32_16_coqa_mi1_0_0 30_2_32_16_coqa_mi1_1_0 30_2_32_16_coqa_mi1_2_0 )
anno_list=(28_0_32_28_coqa_ll2_0_0 28_0_32_28_coqa_ll2_1_0 28_0_32_28_coqa_ll2_2_0 32_0_32_20_coqa_ll3_0_0 32_0_32_20_coqa_ll3_1_0 32_0_32_20_coqa_ll3_2_0 )
gpuid=0

# anno_list=(32_0_32_16_coqa_mi0_0_1 32_0_32_16_coqa_mi0_1_1 32_0_32_16_coqa_mi0_2_1 30_2_32_16_coqa_mi1_0_1 30_2_32_16_coqa_mi1_1_1 30_2_32_16_coqa_mi1_2_1 )
anno_list=(28_0_32_28_coqa_ll2_0_1 28_0_32_28_coqa_ll2_1_1 28_0_32_28_coqa_ll2_2_1 32_0_32_20_coqa_ll3_0_1 32_0_32_20_coqa_ll3_1_1 32_0_32_20_coqa_ll3_2_1 )
gpuid=1

# anno_list=(32_0_32_16_coqa_mi0_0_2 32_0_32_16_coqa_mi0_1_2 32_0_32_16_coqa_mi0_2_2 30_2_32_16_coqa_mi1_0_2 30_2_32_16_coqa_mi1_1_2 30_2_32_16_coqa_mi1_2_2 )
anno_list=(28_0_32_28_coqa_ll2_0_2 28_0_32_28_coqa_ll2_1_2 28_0_32_28_coqa_ll2_2_2 32_0_32_20_coqa_ll3_0_2 32_0_32_20_coqa_ll3_1_2 32_0_32_20_coqa_ll3_2_2 )
gpuid=2

anno_list=(32_32_32_32_coqa_ll_kivi_0_0 32_0_32_32_coqa_ll_asym_0_0 0_16_32_32_coqa_ll_asym_0_0 16_0_32_32_coqa_ll_asym_0_0 )
gpuid=9

echo "numbers of array = ${#anno_list[*]}"

for anno in ${anno_list[@]}
do
    echo $anno

    NUMEXPR_MAX_THREADS='128' CUDA_VISIBLE_DEVICES=$gpuid python run_lm_eval_harness.py --model_name_or_path $model \
        --tasks $tasks \
        --cache_dir ./cached_models \
        --k_bits $bit \
        --v_bits $bit \
        --group_size 32 \
        --residual_length 128 \
        --annotation $anno
    
done