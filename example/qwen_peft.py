from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from chunk_loss_lora.ce import ChunkedCE
from datasets import load_dataset, Dataset
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
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
data = load_dataset('openai/gsm8k', 'main')['train']

model = get_peft_model(model, peft_config)
m = model.lm_head
m_a = model.lm_head.lora_A.default.type(m.weight.dtype)
m_b = model.lm_head.lora_B.default.type(m.weight.dtype)

trainable_parameters = [param for param in model.parameters() if param.requires_grad]
trainer = torch.optim.AdamW(trainable_parameters, lr = 2e-5)

for i in range(100):
    input_ids = tokenizer.apply_chat_template([
        {'role': 'user', 'content': data[0]['question']},
        {'role': 'assistant', 'content': data[0]['answer']}
    ], return_tensors = 'pt').to('cuda')
    o = model(input_ids = input_ids, output_hidden_states = True)

    x = o.hidden_states[-1][:,:-1]
    x.requires_grad = True
    x_ = x.view(-1, x.shape[-1])
    labels_ = input_ids[:,1:].view(-1)
    loss = ChunkedCE.apply(x_, m, m_a, m_b, r, labels_, True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
    trainer.step()

    print(i, loss)

