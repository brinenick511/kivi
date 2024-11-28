model=/new_data/yanghq/models/mistralai/Mistral-7B-Instruct-v0.2
# gpuid=$1
# bit=$2

gpuid=9

anno=16_16_16_16_q_0
bit=16

anno=32_32_32_32_test_3
bit=2

CUDA_VISIBLE_DEVICES=$gpuid python q_pred_long_bench.py --model_name_or_path $model \
    --cache_dir ./cached_models \
    --k_bits $bit \
    --v_bits $bit \
    --k_quant_dim 'channel' \
    --v_quant_dim 'token' \
    --group_size 32 \
    --residual_length 128 \
    --annotation $anno \


python3 eval.py --model_name_or_path $model \
    --cache_dir ./cached_models \
    --k_bits $bit \
    --v_bits $bit \
    --k_quant_dim 'channel' \
    --v_quant_dim 'token' \
    --group_size 32 \
    --residual_length 128 \
    --annotation $anno \


# python3 send.py --model_name_or_path $model \
#     --cache_dir ./cached_models \
#     --k_bits $bit \
#     --v_bits $bit \
#     --k_quant_dim 'channel' \
#     --v_quant_dim 'token' \
#     --group_size 32 \
#     --residual_length 128 \
#     --annotation $anno \

