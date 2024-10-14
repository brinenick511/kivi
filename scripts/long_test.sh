# model e.g.: meta-llama/Llama-2-7b-hf
# ${2:-"model"}
gpuid=$1
k_bits=${2:-"16"}
v_bits=${3:-$k_bits}
group_size=${4:-"32"}
residual_length=${5:-"128"}
model=${6:-"${HOME}/models/mistralai/Mistral-7B-Instruct-v0.2"}
e=0
# if [[ "$model" == ~* ]]; then
#     model="${HOME}${model:1}"
# fi

CUDA_VISIBLE_DEVICES=$gpuid python pred_long_bench.py --model_name_or_path $model \
    --cache_dir ./cached_models \
    --k_bits $k_bits \
    --v_bits $v_bits \
    --group_size $group_size \
    --residual_length $residual_length \
    --e ${e}

CUDA_VISIBLE_DEVICES=$gpuid python3 eval.py --model_name_or_path $model \
    --cache_dir ./cached_models \
    --k_bits $k_bits \
    --v_bits $v_bits \
    --group_size $group_size \
    --residual_length $residual_length \
    --e ${e}

# python3 eval.py --model Mistral-7B-Instruct-v0.2_31500_2bits_group32_residual128
# python3 send.py --model Mistral-7B-Instruct-v0.2_31500_2bits_group32_residual128