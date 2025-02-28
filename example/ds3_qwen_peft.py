"""
deepspeed --master_port=29501 example/ds3_qwen_peft.py
"""

from transformers.integrations import HfDeepSpeedConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2ForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model
from chunk_loss_lora.ce import ChunkedCE
import deepspeed
import torch
import os

MODEL_NAME = os.environ.get('MODEL_NAME', 'Qwen/Qwen2.5-0.5B-Instruct')

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

class Custom(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, linear):
        weight = linear.weight
        ctx.save_for_backward(input, weight)
        ctx.linear = linear
        return linear(input).sum()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        linear = ctx.linear
        linear.weight.grad = torch.zeros_like(weight)
        return torch.zeros_like(input), None

class Model(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, input_ids, labels = None, **kwargs):
        # super_out = super().forward(
        #     input_ids = input_ids,
        #     labels = labels,
        #     output_hidden_states = True,
        # )
        # return {'loss': super_out.loss}

        super_out = super().forward(
            input_ids = input_ids,
            output_hidden_states = True,
        )
        x = super_out.hidden_states[-1]
        x.requires_grad = True
        x_ = x.view(-1, x.shape[-1])
        labels = labels.view(-1)
        m = self.lm_head
        m_a = self.lm_head.lora_A.default
        m_b = self.lm_head.lora_B.default
        loss = ChunkedCE.apply(x_, m, m_a, m_b, r, labels, True)
        return {'loss': loss}

model = Model.from_pretrained(
    MODEL_NAME,
    torch_dtype = torch.bfloat16
)
model = get_peft_model(model, peft_config)
model_engine, optimizer, _, _ = deepspeed.initialize(model=model, config_params=ds_config)
print('weight is deepspeed', hasattr(model.lm_head.weight, 'ds_param_type'))

input_ids = torch.tensor([1,2,3])[None].cuda()

o = model(input_ids = input_ids, labels = input_ids)
print(o)
model_engine.backward(o['loss'])
model_engine.step()
