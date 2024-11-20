gpuid=$1
bit=${2:-"2"}
# kv=${3:-"kv"}
# id_l=${4:-"16"}
# id_u=${5:-"32"}
model=/new_data/yanghq/models/mistralai/Mistral-7B-Instruct-v0.2
anno_list=(${bit}_k_16_32 ${bit}_v_16_32 ${bit}_kv_16_32 )
anno_list=(${bit}_v_12_32)
anno_list=(${bit}_30_32_16_32)
anno_list=(2_32_32_16_32_0 2_32_32_14_32_0 2_32_32_12_32_0 2_30_32_16_32_0 2_30_32_14_32_0 2_30_32_12_32_0 2_32_32_16_32_1 2_32_32_14_32_1 2_32_32_12_32_1 2_30_32_16_32_1 2_30_32_14_32_1 2_30_32_12_32_1 )
anno_list=(2_32_32_16_32_2 2_32_32_14_32_2 2_32_32_12_32_2 2_30_32_16_32_2 2_30_32_14_32_2 2_30_32_12_32_2 2_32_32_16_32_3 2_32_32_14_32_3 2_32_32_12_32_3 2_30_32_16_32_3 2_30_32_14_32_3 2_30_32_12_32_3 )
anno_list=(2_32_32_16_32_4 2_32_32_14_32_4 2_32_32_12_32_4 2_30_32_16_32_4 2_30_32_14_32_4 2_30_32_12_32_4 2_32_32_16_32_5 2_32_32_14_32_5 2_32_32_12_32_5 2_30_32_16_32_5 2_30_32_14_32_5 2_30_32_12_32_5 )
anno_list=(2_30_32_16_32_3 2_32_32_12_32_3 2_30_32_14_32_3)
anno_list=(2_30_32_16_32_10 2_30_32_16_32_11 2_30_32_16_32_12 2_30_32_16_32_13 2_30_32_16_32_14 2_30_32_16_32_15 )
# anno_list=(2_30_32_16_32_10 2_30_32_16_32_11 2_30_32_16_32_12 2_30_32_16_32_13 2_30_32_16_32_14 2_30_32_16_32_15 2_32_32_14_32_10 2_32_32_14_32_11 2_32_32_14_32_12 2_32_32_14_32_13 2_32_32_14_32_14 2_32_32_14_32_15 )
anno_list=(2_30_32_12_32_10 )
anno_list=(2_0_0_0_0_0 )
anno_list=(${bit}_0_0_0_0_0 )
anno_list=(${bit}_30_32_16_32 )



echo "numbers of array = ${#anno_list[*]}"

for anno in ${anno_list[@]}
do
    echo $anno
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$gpuid python v_pred_long_bench.py --model_name_or_path $model \
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
