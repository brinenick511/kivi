# This script is adapted from
# https://github.com/FranxYao/Long-Context-Data-Engineering.git
gpuid=9

mkdir -p ./results_needle/logs/
mkdir -p ./results_needle/img/
MAX_CAPACITY_PROMPT=96  # [64, 96, 128, 256, 512, 1024, 2048, ...]
attn_implementation="flash_attention_2" # Support "flash_attention_2", "sdpa", "None".
MODEL=/new_data/yanghq/models/mistralai/Mistral-7B-Instruct-v0.2
MODEL=/new_data/yanghq/models/mistralai/Mistral-7B-v0.3

# ['full', 'pyramidkv', 'snapkv', 'streamingllm', 'h2o', 'cam']
METHOD='full'
METHOD='ours'

TAG=test


STEP=2550
MIN_LEN=1000
MAX_LEN=26501
anno=32_32_32_32_kivi_0_0
anno=16_16_16_16_mibl_0_0
anno=32_0_32_32_asym_0_0


# For Llama3-8b

# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 8001\
#     --model_provider LLaMA3 \
#     --model_name YOU_PATH_TO_LLAMA_3 \
#     --attn_implementation ${attn_implementation} \
#     --step 100 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version LlaMA3_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/LlaMA3_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log


# For Mistral

(
CUDA_VISIBLE_DEVICES=$gpuid python -u run_needle_in_haystack.py --s_len $MIN_LEN --e_len $MAX_LEN \
    --model_provider Mistral \
    --model_name $MODEL \
    --step $STEP \
    --method $METHOD \
    --max_capacity_prompt $MAX_CAPACITY_PROMPT \
    --model_version Mistral2_${METHOD}_${anno} \
    --annotation $anno
) 2>&1  | tee ./results_needle/logs/Mistral2_${METHOD}_${anno}.log

python ./scripts/scripts_needle/visualize.py --result_path ./results_needle/results/Mistral2_${METHOD}_${anno}/
