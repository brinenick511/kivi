gpuid=$1
bit=$2
kv=${3:-"test"}

# kv=${3:-"kv"}
# id_l=${4:-"16"}
# id_u=${5:-"32"}
model=/new_data/yanghq/models/mistralai/Mistral-7B-Instruct-v0.2
# anno=${bit}_kv_0_32
# anno=${bit}_k_16_32
# anno=${bit}_k_14_30
anno=${bit}_${kv}
# anno=${bit}_${kv}_${id_l}_${id_u}
# echo $anno

CUDA_VISIBLE_DEVICES=$gpuid python pred_long_bench.py --model_name_or_path $model \
    --cache_dir ./cached_models \
    --k_bits $bit \
    --v_bits $bit \
    --k_quant_dim 'channel' \
    --v_quant_dim 'token' \
    --group_size 32 \
    --residual_length 128 \
    --annotation $anno \


# python3 eval.py --model_name_or_path $model \
#     --cache_dir ./cached_models \
#     --k_bits $bit \
#     --v_bits $bit \
#     --k_quant_dim 'channel' \
#     --v_quant_dim 'token' \
#     --group_size 32 \
#     --residual_length 128 \
#     --annotation $anno \


# python3 send.py --model_name_or_path $model \
#     --cache_dir ./cached_models \
#     --k_bits $bit \
#     --v_bits $bit \
#     --k_quant_dim 'channel' \
#     --v_quant_dim 'token' \
#     --group_size 32 \
#     --residual_length 128 \
#     --annotation $anno \

