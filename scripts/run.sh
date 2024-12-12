model=/new_data/yanghq/models/mistralai/Mistral-7B-Instruct-v0.2
model=/new_data/yanghq/models/meta-llama/Llama-2-7b-chat-hf
# gpuid=$1
# bit=$2

gpuid=0

anno=16_16_16_16_bl_0_0
bit=16

# anno=26_26_32_32_test_0_0
# anno=30_16_32_32_test_0_0
# # anno=28_30_30_28_test_1_1
# anno=32_0_32_32_asym_1_1
# anno=28_30_30_28_test_ll_1_1
# anno=32_0_32_32_asym_0_0
anno=32_32_32_32_kivi_0_0

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


python3 send.py --model_name_or_path $model \
    --cache_dir ./cached_models \
    --k_bits $bit \
    --v_bits $bit \
    --k_quant_dim 'channel' \
    --v_quant_dim 'token' \
    --group_size 32 \
    --residual_length 128 \
    --annotation $anno \

