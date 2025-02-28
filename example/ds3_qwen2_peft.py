"""
deepspeed --master_port=29501 example/ds3_qwen2_peft.py
"""
from transformers.integrations import HfDeepSpeedConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model
from chunk_loss_lora.ce import ChunkedCE
import deepspeed
import torch
import os

MODEL_NAME = os.environ.get('MODEL_NAME', 'Qwen/Qwen2.5-0.5B-Instruct')

config = AutoConfig.from_pretrained(MODEL_NAME)
model_hidden_size = config.hidden_size

ds_config = {
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": int(0.9 * model_hidden_size * model_hidden_size),
        "stage3_param_persistence_threshold": 10 * model_hidden_size
    },
    "steps_per_print": 2000,
    "train_batch_size": 2,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
}
dschf = HfDeepSpeedConfig(ds_config)

rank = 64
alpha = rank * 2
r = alpha / rank
peft_config = LoraConfig(
    lora_alpha=alpha,
    lora_dropout=0.0,
    r=rank,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["lm_head"],
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype = torch.bfloat16
)

model = get_peft_model(model, peft_config)

ds_engine = deepspeed.initialize(model=model, config_params=ds_config)
print('weight is deepspeed', hasattr(model.lm_head.weight, 'ds_param_type'))

input_ids = torch.tensor([1,2,3])[None].cuda()
o = model(input_ids = input_ids, output_hidden_states = True)
m = model.lm_head
m_a = model.lm_head.lora_A.default
m_b = model.lm_head.lora_B.default

x = o.hidden_states[-1]
x_ = x.view(-1, x.shape[-1])
labels_ = input_ids.view(-1)
loss = ChunkedCE.apply(x_.type(torch.float32), m.weight.type(torch.float32), m_a.weight, m_b.weight, r, labels_, True)
print(loss)
