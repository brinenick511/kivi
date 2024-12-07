# LLaMA model with KIVI
import torch
import os
from models.llama_kivi import LlamaForCausalLM_KIVI
from models.r_mistral_kivi import MistralForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer, MistralConfig
import time
import sys
import logging
logging.basicConfig(
    filename='/new_data/yanghq/tm.log',  # 日志文件名
    level=logging.INFO, 
    format='%(message)s'
)

# logging.info("日志已写入文件")
# exit(0)

K_BITS = 2
V_BITS = 2
GROUP_SIZE = 32
RESIDUAL_LENGTH = 128
BATCH_SIZE = 16
PATH_TO_YOUR_SAVE_DIR = './outputs'
ANNOTATION = '32_32_32_32_0_0'
assert len(sys.argv) > 1, f'len(sys.argv) = {len(sys.argv)}'
ANNOTATION = str(sys.argv[-1]).strip()
logging.info(ANNOTATION)

model_name_or_path = 'meta-llama/Llama-2-7b-hf'
model_name_or_path = '/new_data/yanghq/models/mistralai/Mistral-7B-Instruct-v0.2'
config = MistralConfig.from_pretrained(model_name_or_path)
config.k_bits = K_BITS # current support 2/4 bit for KV Cache
config.v_bits = V_BITS # current support 2/4 bit for KV Cache
config.group_size = GROUP_SIZE
config.residual_length = RESIDUAL_LENGTH # the number of recent fp16 tokens
CACHE_DIR = PATH_TO_YOUR_SAVE_DIR

if K_BITS < 16 and V_BITS < 16:
    config.use_flash = True
    config.annotation = str(ANNOTATION).strip()
    model = MistralForCausalLM_KIVI.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
else:
    from transformers import MistralForCausalLM
    model = MistralForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, 
    use_fast=False, 
    trust_remote_code=True, 
    tokenizer_type='llama')

model.cuda().eval()

context = []
batch_size = BATCH_SIZE
prompt_lenth = 256
output_length = 1024*4
num_repeats = 3
for _ in range(batch_size):
    string = 't,' * (prompt_lenth // 2)
    context.append(string[:-1])
inputs = tokenizer(context, return_tensors="pt").to('cuda')
input_ids = inputs['input_ids']
# print(f"bs: {batch_size}, seqlen: {input_ids.shape[1]}+{output_length}\nmodel:{model_name_or_path}")
print(f"\nbs: {batch_size}, seqlen: {input_ids.shape[1]}+{output_length}\n")
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    torch.cuda.synchronize()
    st = time.time()
    for i in range(num_repeats):
        outputs = model.generate(**inputs, max_new_tokens=output_length)
    torch.cuda.synchronize()
    t = (time.time() - st) / num_repeats * 1000
    t=int(t)
    print(f'used time: {t} ms')
    used_mem = torch.cuda.max_memory_allocated()
    m = used_mem / 1024 ** 2
    m=int(m)
    print(f'peak mem: {m} MB')
    logging.info(f'{t}\t{m}')