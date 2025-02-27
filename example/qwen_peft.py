from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from chunk_loss_lora.ce import ChunkedCE
import torch

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
    'Qwen/Qwen2.5-0.5B-Instruct',
    torch_dtype = torch.bfloat16
).cuda()

model = get_peft_model(model, peft_config)
input_ids = torch.tensor([1,2,3])[None].cuda()
o = model(input_ids = input_ids, output_hidden_states = True)
m = model.lm_head
m_a = model.lm_head.lora_A.default
m_b = model.lm_head.lora_B.default

x = o.hidden_states[-1]
x_ = x.view(-1, x.shape[-1])
labels_ = input_ids.view(-1)
loss = ChunkedCE.apply(x_.type(torch.float32), m.weight.type(torch.float32), m_a.weight, m_b.weight, r, labels_, True)
loss.backward()
print(m_a.weight.grad)
print(m_b.weight.grad)
