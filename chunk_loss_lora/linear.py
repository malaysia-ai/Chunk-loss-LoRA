import os
import torch
import triton
import triton.language as tl

CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', '32'))

@triton.jit
def lora_matmul_kernel(
        input_ptr, w_ptr, w1_ptr, w2_ptr, output_ptr,
        M, N, K, R, ALPHA,
        stride_im, stride_ik,
        stride_wn, stride_wk,
        stride_w1r, stride_w1k,
        stride_w2n, stride_w2r,
        stride_om, stride_on,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, 
        BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_R: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_r = tl.arange(0, BLOCK_SIZE_R)
    
    input_ptrs = input_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik)
    w_ptrs = w_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)
    
    accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        i = tl.load(input_ptrs)
        w = tl.load(w_ptrs)
        accum = tl.dot(i, w, accum)

        input_ptrs += BLOCK_SIZE_K * stride_ik
        w_ptrs += BLOCK_SIZE_K * stride_wk
    
    input_ptrs = input_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik)
    w1_ptrs = w1_ptr + (offs_r[None, :] * stride_w1r + offs_k[:, None] * stride_w1k)
    
    temp = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_R), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        i = tl.load(input_ptrs)
        w1 = tl.load(w1_ptrs)

        temp = tl.dot(i, w1, temp)
        
        input_ptrs += BLOCK_SIZE_K * stride_ik
        w1_ptrs += BLOCK_SIZE_K * stride_w1k
    
    w2_ptrs = w2_ptr + (offs_n[None, :] * stride_w2n + offs_r[:, None] * stride_w2r)
    mask_w2 = (offs_n[None, :] < N) & (offs_r[:, None] < R)
    w2 = tl.load(w2_ptrs, mask=mask_w2, other=0.0)
    lora_acc = tl.dot(temp.to(w2.dtype), w2)
    
    output = accum + lora_acc * ALPHA
    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, output, mask=mask_out)

def matmul_chunk(input_chunk, weight, weight_a, weight_b, r):
    left = input_chunk @ weight
    right = (input_chunk @ weight_a) @ weight_b
    return left + right * r

class LinearLora(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, _input, m, m_a, m_b, r, 
        compiled=True, 
        CHUNK_SIZE=CHUNK_SIZE, 
    ):
        B, L, D = _input.shape
        N, _ = m.shape
        _input = _input.reshape(-1, D)
        chunks = max(_input.shape[0] // CHUNK_SIZE, 1)
        outputs = []
        def accumulate_chunk(input_chunk):
            return matmul_chunk(input_chunk, m.T, m_a.T, m_b.T, r)
        
        if compiled:
            accumulate_chunk = torch.compile(accumulate_chunk)
        
        input_chunks = torch.chunk(_input, chunks=chunks, dim=0)
        for input_chunk in input_chunks:
            outputs.append(accumulate_chunk(input_chunk))
        
        outputs = torch.cat(outputs, dim=0)
        ctx.save_for_backward(
            _input, m, m_a, m_b,
        )
        ctx.compiled = compiled
        ctx.CHUNK_SIZE = CHUNK_SIZE
        return outputs.reshape(B, L, -1)
    
    @staticmethod
    def backward(ctx, grad_output):
        _input, m, m_a, m_b = ctx.saved_tensors
        compiled = ctx.compiled
        CHUNK_SIZE = ctx.CHUNK_SIZE

        B, L, D = grad_output.shape
        N, _ = m.shape
        grad_output = grad_output.reshape(-1, D)
        chunks = max(grad_output.shape[0] // CHUNK_SIZE, 1)
        outputs = []
        def accumulate_chunk(input_chunk):
            return matmul_chunk(input_chunk, m, m_b, m_a, 1.0)
        
        if compiled:
            accumulate_chunk = torch.compile(accumulate_chunk)
        
        input_chunks = torch.chunk(grad_output, chunks=chunks, dim=0)
        for input_chunk in input_chunks:
            outputs.append(accumulate_chunk(input_chunk))
        
        outputs = torch.cat(outputs, dim=0)
        grad_input = outputs.reshape(B, L, -1)
        
        """
        # Gradient for x
        grad_x = grad_output @ m.T + (grad_output @ b.T) @ a.T
        
        # Gradient for m
        grad_m = x.T @ grad_output
        
        # Gradient for a
        grad_a = x.T @ (grad_output @ b.T)
        
        # Gradient for b
        grad_b = xa.T @ grad_output
        """

        return (grad_input, None, torch.zeros_like(m_a), torch.zeros_like(m_b), None, None)


