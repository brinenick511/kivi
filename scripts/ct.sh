gpuid=$1
bit=$2
# kv=${3:-"kv"}
# id_l=${4:-"16"}
# id_u=${5:-"32"}
model=/new_data/yanghq/models/mistralai/Mistral-7B-Instruct-v0.2
anno_list=(${bit}_k_0_16 ${bit}_k_8_24 ${bit}_k_16_32 ${bit}_v_0_16 ${bit}_v_8_24 ${bit}_v_16_32 ${bit}_kv_0_16 ${bit}_kv_8_24 ${bit}_kv_16_32 )
anno_list=(${bit}_k_0_16 ${bit}_k_8_24 ${bit}_k_16_32 ${bit}_v_0_16 ${bit}_v_8_24 ${bit}_v_16_32 ${bit}_kv_0_16 ${bit}_kv_8_24 ${bit}_kv_16_32 )
anno_list=(${bit}_k_0_16 ${bit}_k_16_32 ${bit}_k_0_32 ${bit}_k_24_32 ${bit}_v_0_16 ${bit}_v_16_32 ${bit}_v_0_32 ${bit}_v_24_32 ${bit}_kv_0_16 ${bit}_kv_16_32 ${bit}_kv_0_32 ${bit}_kv_24_32 )
anno_list=(${bit}_k_8_32 ${bit}_k_8_32 ${bit}_v_8_32 ${bit}_v_8_32 ${bit}_kv_8_32 ${bit}_kv_8_32 )
anno_list=(${bit}_v_16_32 ${bit}_k_16_32 ${bit}_kv_16_32 )


echo "numbers of array = ${#anno_list[*]}"

for anno in ${anno_list[@]}
do
    echo $anno
    CUDA_VISIBLE_DEVICES=$gpuid python v_pred_long_bench.py --model_name_or_path $model \
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
