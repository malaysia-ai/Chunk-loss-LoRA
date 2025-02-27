import torch
import torch.nn as nn
from chunk_loss_lora.ce import ChunkedCE
from cut_cross_entropy import linear_cross_entropy

torch.set_default_device('cuda')

B, T, D, V, R = 4, 1024, 768, 128256, 64
r = 2.0
m = nn.Linear(D, V, bias = False).to(torch.bfloat16)
m.weight.requires_grad = False
m_a = nn.Linear(D, R, bias = False).to(torch.bfloat16)
m_b = nn.Linear(R, V, bias = False).to(torch.bfloat16)
ce = nn.CrossEntropyLoss()
x = torch.randn(B, T, D, requires_grad=True, dtype=torch.bfloat16)
label = torch.randint(0, V, (B, T)).to(torch.int64)

def f(x, label):
    logits = m(x) + m_b(m_a(x)) * r
    out = ce(logits.view(-1, V), label.view(-1))
    out.backward()
    return out

def chunked_f(x, label, compiled=False):
    out = ChunkedCE.apply(x.view(-1, D), m.weight, m_a.weight, m_b.weight, r, label.view(-1), compiled)
    out.backward()
    return out

def bench(f, name=None, iters=100, warmup=5, display=True, profile=False, profile_mem=False):
    import time
    from triton.testing import do_bench
    for _ in range(warmup):
        f()

    if profile_mem:
        torch.cuda.memory._record_memory_history()
        f()
        torch.cuda.memory._dump_snapshot(f"{name if name is not None else 'memory'}.pickle")
    if profile:
        with torch.profiler.profile() as prof:
            f()
        prof.export_chrome_trace(f"{name if name is not None else 'trace'}.json")

    torch.cuda.reset_peak_memory_stats()
    ms_per_iter = do_bench(lambda: f())
    if name is None:
        res = ms_per_iter
    else:
        res= f"{name}: {ms_per_iter:.3f}ms"
    if display:
        print(res)
        print("Peak mem: ", torch.cuda.max_memory_allocated()/1e9)
        print()
    return res

opt_f = torch.compile(f)
bench(lambda: f(x, label), name='eager (non-chunked)')
bench(lambda: chunked_f(x, label, compiled=False), name='eager (chunked)')
bench(lambda: opt_f(x, label), name='compile (non-chunked)')
bench(lambda: chunked_f(x, label, compiled=True), name='compile (chunked)')

"""
eager (non-chunked): 46.425ms
Peak mem:  4.731470848

eager (chunked): 68.193ms
Peak mem:  2.388297216

compile (non-chunked): 39.854ms
Peak mem:  2.646541312

compile (chunked): 40.364ms
Peak mem:  1.600300032
"""