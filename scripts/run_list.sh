gpuid=$1
bit=${2:-"2"}
# kv=${3:-"kv"}
# id_l=${4:-"16"}
# id_u=${5:-"32"}
model=/new_data/yanghq/models/mistralai/Mistral-7B-Instruct-v0.2
anno_list=(32_32_32_32_c_0 32_32_32_32_c_1 32_32_32_32_c_2 32_32_32_32_c_3 )
anno_list=(32_32_32_32_kivi_0_0 32_16_32_32_asym_0_0)


echo "numbers of array = ${#anno_list[*]}"

for anno in ${anno_list[@]}
do
    echo $anno
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$gpuid python q_pred_long_bench.py --model_name_or_path $model \
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
