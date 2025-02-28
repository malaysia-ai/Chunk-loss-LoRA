# Chunk-loss-LoRA

Fused kernel chunk loss to include LoRA to reduce memory, support DeepSpeed ZeRO3.

1. Support DeepSpeed Zero3, check [monkey patched](chunk_loss_lora/__init__.py).
3. Support PyTorch chunking Torch compile and Triton.
2. Will wrap HuggingFace PEFT Forward properly.

## Examples

1. [example/ds3_qwen_peft.py](example/ds3_qwen_peft.py), simple DeepSpeed ZeRO3 training loop.
2. [example/hf_trainer.py](example/hf_trainer.py), train using HuggingFace Trainer.
3. [Example/qwen_peft.py](example/qwen_peft.py), simple training loop without DeepSpeed.

Currently only [chunk_loss_lora.ce.ChunkedCE](chunk_loss_lora/ce.py) been optimized for weird cases DeepSpeed ZeRO3.

## Benchmarks

All the benchmark is based on default `CHUNK_SIZE` 32.

### Fused Cross Entropy

Based on [benchmark/ce.py](benchmark/ce.py),

```
liger lce: 128.493ms
Peak mem:  0.567181824

eager (non-chunked): 45.947ms
Peak mem:  4.71505408

eager (chunked): 133.431ms
Peak mem:  0.600466944

compile (non-chunked): 39.857ms
Peak mem:  2.630124544

compile (chunked): 104.765ms
Peak mem:  0.567433216

compile (chunked module): 105.556ms
Peak mem:  0.583849984
```

## Special thanks

We evolve from https://gist.github.com/Chillee/22cd93e11b887db1f596ab754d60a899