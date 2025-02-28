# Chunk-loss-LoRA

Chunk loss for LoRA.

1. Support DeepSpeed Zero3.
3. Support PyTorch chunking Torch compile and Triton.
2. Will wrap HuggingFace PEFT Forward properly.

## Benchmarks

All the benchmark is based on default `CHUNK_SIZE` 32.

### Fused Cross Entropy

Based on [benchmark/ce.py](benchmark/ce.py),

```
liger lce: 128.683ms
Peak mem:  0.567181824

eager (non-chunked): 45.930ms
Peak mem:  4.71505408

eager (chunked): 133.092ms
Peak mem:  0.600466944

compile (non-chunked): 39.164ms
Peak mem:  2.630124544

compile (chunked): 105.000ms
Peak mem:  0.567433216
```