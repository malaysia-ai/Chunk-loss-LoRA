"""
deepspeed --master_port=29501 example/ds3_qwen_peft.py
"""

from transformers.integrations import HfDeepSpeedConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2ForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model
from chunk_loss_lora.ce import ChunkedCE
from datasets import load_dataset
import deepspeed
import torch
import os

MODEL_NAME = os.environ.get('MODEL_NAME', 'Qwen/Qwen2.5-0.5B-Instruct')
STAGE = int(os.environ.get('STAGE', '3'))

config = AutoConfig.from_pretrained(MODEL_NAME)
model_hidden_size = config.hidden_size

ds_config = {
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
        }
    },
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": True
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
    "steps_per_print": 1,
    "gradient_clipping": 1.0,
    "train_batch_size": 1,
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

class Model(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, input_ids, labels = None, **kwargs):
        super_out = super().forward(
            input_ids = input_ids,
            output_hidden_states = True,
        )
        x = super_out.hidden_states[-1][:,:-1]
        x.requires_grad = True
        x_ = x.view(-1, x.shape[-1])
        labels = labels[:,1:].view(-1)
        m = self.lm_head
        m_a = self.lm_head.lora_A.default
        m_b = self.lm_head.lora_B.default
        loss = ChunkedCE.apply(x_, m.weight, m_a.weight, m_b.weight, r, labels, True)
        return {'loss': loss}

model = Model.from_pretrained(
    MODEL_NAME,
    torch_dtype = torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
data = load_dataset('openai/gsm8k', 'main')['train']

model = get_peft_model(model, peft_config)
m_a = model.lm_head.lora_A.default
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model, 
    model_parameters=model.parameters(),
    config_params=ds_config
)
print('weight is deepspeed', hasattr(model.lm_head.weight, 'ds_param_type'))
model_engine.train()
for i in range(100):
    input_ids = tokenizer.apply_chat_template([
        {'role': 'user', 'content': data[0]['question']},
        {'role': 'assistant', 'content': data[0]['answer']}
    ], return_tensors = 'pt').to('cuda')
    o = model_engine(input_ids = input_ids, labels = input_ids)
    model_engine.backward(o['loss'])
    print(i, o['loss'])

    model_engine.step()
    model_engine.zero_grad()
