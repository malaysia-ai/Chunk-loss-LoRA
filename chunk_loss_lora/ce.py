import os
import torch
import torch.nn as nn
import triton
import triton.language as tl
from contextlib import nullcontext
try:
    import deepspeed
    from deepspeed.utils import safe_set_full_grad
except:
    deepspeed = None

CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', '32'))
ce = nn.CrossEntropyLoss()

@triton.jit
def liger_cross_entropy_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore,
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This kernel computes both cross entropy loss and the gradient of the input.
    We only consider hard label + mean reduction for now. Please refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.
    Parameters:
    X_ptr: Pointer to input tensor.
    X_stride (int): The stride of the input tensor.
    Y_ptr: Pointer to target tensor.
    Y_stride (int): The stride of the target tensor.
    loss_ptr: Pointer to tensor to store the loss.
    loss_stride (int): The stride of the loss tensor.
    n_cols (int): The number of columns in the input tensor.
    n_non_ignore (int): The number of non-ignored elements in the batch.
    ignore_index (int): The index to ignore in the target.
    BLOCK_SIZE (int): The block size for Triton operations.
    """

    # https://github.com/triton-lang/triton/issues/1058
    # If B*T*V is too large, program_id * stride will overflow out of int32, so we convert to int64
    program_id = tl.program_id(0).to(tl.int64)

    # 1. Load Y_ptr first because if the target is ignore_index, we can return right away
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    # 2. locate the start index
    X_ptr += program_id * X_stride

    if y == ignore_index:
        # set all X_ptr as 0
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    loss_ptr += program_id * loss_stride

    # Online softmax: 2 loads + 1 store (compared with 3 loads + 1 store for the safe softmax)
    # Refer to Algorithm 3 in the paper: https://arxiv.org/pdf/1805.02867

    # 3. [Online softmax] first pass: find max + sum
    m = float("-inf")  # m is the max value. use the notation from the paper
    d = 0.0  # d is the sum. use the notation from the paper
    ori_X_y = tl.load(
        X_ptr + y
    )  # we need to store the original value of X_y for the loss calculation

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")
        )
        block_max = tl.max(X_block)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    # 4. [Online softmax] second pass: calculate the gradients
    # dx_y = (softmax(x_y) - 1) / N
    # dx_i = softmax(x_i) / N, i != y
    # N is the number of non ignored elements in the batch
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")
        )
        X_block = (tl.exp(X_block - m) / d) / (n_non_ignore)
        tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)

    # We need tl.debug_barrier() to ensure the new result of X_ptr is written as mentioned in
    # https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/ops/cross_entropy.py#L34
    tl.debug_barrier()

    # 5. Calculate the loss

    # loss = log (softmax(X_y)) = log ((e ^ (X_y - max(X)) / sum(e ^ (X - max(X))))
    #      = (X_y - max(X)) - log(sum(e ^ (X - max(X))))
    # sum(e ^ (X - max(X))) must >= 1 because the max term is e ^ 0 = 1
    # So we can safely calculate log (softmax(X_y)) without overflow
    loss = -(ori_X_y - m - tl.log(d))

    # 6. Specially handle the i==y case where `dx_y = (softmax(x_y) - 1) / N`
    X_y = tl.load(X_ptr + y)
    X_y += -1 / (n_non_ignore)

    tl.store(loss_ptr, loss)
    tl.store(X_ptr + y, X_y)


# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576 https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536 // 2  # the best size we found by manually tuning


@triton.jit
def element_mul_kernel(
    X_ptr,
    X_stride,
    grad_output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This function multiplies each element of the tensor pointed by X_ptr with the value pointed by grad_output_ptr.
    The multiplication is performed in-place on the tensor pointed by X_ptr.
    Parameters:
    X_ptr: Pointer to the input tensor.
    X_stride (int): The stride of the input tensor.
    grad_output_ptr: Pointer to the gradient output value.
    n_cols (int): The number of columns in the input tensor.
    BLOCK_SIZE (int): The block size for Triton operations.
    """

    # Get the program ID and convert it to int64 to avoid overflow
    program_id = tl.program_id(0).to(tl.int64)

    # Locate the start index
    X_ptr += program_id * X_stride

    # Load the gradient output value
    grad_output = tl.load(grad_output_ptr)

    # Perform the element-wise multiplication
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(X_ptr + X_offsets, X_block * grad_output, mask=X_offsets < n_cols)



# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576 https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536 // 2


def fused_linear_cross_entropy_forward(
    _input, weight, weight_a, weight_b, r, target, ignore_index=-100
):
    dtype = (
        torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else _input.dtype
    )
    device = _input.device

    # inputs have shape: BT x H
    # materialized activations will have shape: BT x V
    # the increase in memory = BT x V
    # reduction can be achieved by partitioning the number of tokens BT into smaller chunks.
    # for ex: if we were to achieve the same memory consumption as BT x H, then the chunk size should be:
    # inc_factor = (V+H-1)//H, chunk_size = (BT + inc_factor - 1)//inc_factor
    # for ex: BT = 4096*4, V = 32000, H = 4096 ==> inc_factor = 8, chunk_size = 2048
    BT, H = _input.shape
    V = weight.shape[0]
    R = weight_a.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    inc_factor = triton.cdiv(V, H)  # (V + H - 1) // H
    chunk_size = triton.next_power_of_2(
        triton.cdiv(BT, inc_factor)
    )  # (BT + inc_factor - 1) // inc_factor
    num_chunks = triton.cdiv(BT, chunk_size)  # (BT + chunk_size - 1) // chunk_size

    grad_weight_a = torch.zeros_like(weight_a, device=device)
    grad_weight_b = torch.zeros_like(weight_b, device=device)
    grad_input = torch.zeros_like(_input, device=device)
    # we use fp32 for loss accumulator
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)

    total_n_non_ignore = (target != ignore_index).sum()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]  # chunk_size x H

        # when doing matmul, use the original precision
        logits_chunk = _input_chunk @ weight.t()  # chunk_size x V
        right_chunk = (_input_chunk @ weight_a.t()) @ weight_b.t()
        logits_chunk = logits_chunk + right_chunk * r
        target_chunk = target[start_idx:end_idx]  # chunk_size,

        n_rows = logits_chunk.shape[0]

        # unreduced loss
        loss_1d_slice = loss_1d[start_idx:end_idx]  # chunk_size,
        n_non_ignore = (target_chunk != ignore_index).sum().item()

        # when doing CE, use the upcasted precision
        logits_chunk = logits_chunk.float()

        # ensure _input and target are contiguous
        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        # Here we calculate the gradient of logits_chunk in place so we can save memory.
        liger_cross_entropy_kernel[(n_rows,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),  # always 1
            loss_ptr=loss_1d_slice,
            loss_stride=loss_1d_slice.stride(-1),  # always 1
            n_cols=V,
            n_non_ignore=n_non_ignore,
            ignore_index=ignore_index,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )

        # gradient of logits_chunk is computed in-place by the above triton kernel.
        # Following HuggingFace model source code, we do the forward and backward
        # w.r.t. logits in fp32 for numerical stability especially as the num classes (vocab size) os huge.
        # (reference: https://github.com/huggingface/transformers/blob/v4.42.4/src/transformers/models/llama/modeling_llama.py#L1194)
        # Propagating to lm_head's backward, we'll switch back to the original dtype.
        logits_chunk = logits_chunk.to(dtype)

        # gradient of logits_chunk is computed in-place by the above triton kernel and is of shape: chunk_size x V
        # thus grad_input[start_idx: end_idx] should be of shape: chunk_size x H
        # additionally, since we are chunking the inputs, observe that the loss and gradients are calculated only
        # on `n_non_ignore` tokens. However, the gradient of the input should be calculated for all tokens.
        # Thus, we need an additional scaling factor of (n_non_ignore/total_n_non_ignore) to scale the gradients.
        grad_logits_chunk = logits_chunk * (
            n_non_ignore / total_n_non_ignore
        )  # chunk_size x V
        # grad_logits_chunk = [chunk, V]
        # _input_chunk = [chunk, D]
        grad_input[start_idx:end_idx] = grad_logits_chunk @ weight + (grad_logits_chunk @ weight_b) @ weight_a

        # [R, D] = ([R, V] x [V, C]) x [C, D]
        torch.addmm(
            input=grad_weight_a,
            mat1=weight_b.t() @ logits_chunk.t(),
            mat2=_input_chunk,
            out=grad_weight_a,
            alpha=n_non_ignore / total_n_non_ignore,
            beta=1.0,
        )

        # (X @ W_A.T) @ grad_output
        # [V, R] = [V, C] x ([C, D] x [D, R])
        torch.addmm(
            input=grad_weight_b,
            mat1=logits_chunk.t(),
            mat2=(_input_chunk @ weight_a.t()),
            out=grad_weight_b,
            alpha=n_non_ignore / total_n_non_ignore,
            beta=1.0,
        )

    loss = torch.sum(loss_1d) / total_n_non_ignore
    return loss, grad_input, grad_weight_a, grad_weight_b


class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _input, weight, weight_a, weight_b, r, target, ignore_index=-100):
        loss, grad_input, grad_weight_a, grad_weight_b = fused_linear_cross_entropy_forward(
            _input, weight, weight_a, weight_b, r, target, ignore_index
        )
        ctx.save_for_backward(
            grad_input.detach(),
            grad_weight_a.detach(),
            weight_b.detach(),
        )
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (grad_input, grad_weight_a, grad_weight_b) = ctx.saved_tensors
        return (grad_input, None, grad_weight_a, grad_weight_b, None, None, None, None)

class ChunkedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _input, m, m_a, m_b, r, target, compiled=True, CHUNK_SIZE=CHUNK_SIZE):
        
        is_module = False
        is_deepspeed = False
        if isinstance(m, nn.Module):
            weight = m.weight
            weight_a = m_a.weight
            weight_b = m_b.weight
            is_module = True
        else:
            weight = m
            weight_a = m_a
            weight_b = m_b
            
        if hasattr(weight, 'ds_param_type'):
            gather_weight = deepspeed.zero.GatheredParameters(weight)
            gather_weight_a = deepspeed.zero.GatheredParameters(weight_a)
            gather_weight_b = deepspeed.zero.GatheredParameters(weight_b)
            is_deepspeed = True
        else:
            gather_weight = nullcontext()
            gather_weight_a = nullcontext()
            gather_weight_b = nullcontext()
        
        with gather_weight, gather_weight_a, gather_weight_b:

            def compute_loss(input_chunk, weight, weight_a, weight_b, r, target):
                left = input_chunk @ weight.T
                right = (input_chunk @ weight_a.T) @ weight_b.T
                logits = left + right * r
            
                logits = logits.float()
                loss = ce(logits, target)
                return loss

            grad_weight_a = torch.zeros_like(weight_a)
            grad_weight_b = torch.zeros_like(weight_b)
            grad_inputs = []
            loss_acc = torch.zeros((), device=_input.device)

            chunks = max(_input.shape[0] // CHUNK_SIZE, 1)
            def accumulate_chunk(input_chunk, target_chunk):
                (chunk_grad_input, chunk_grad_weight_a, chunk_grad_weight_b), chunk_loss = torch.func.grad_and_value(compute_loss, argnums=(0,2,3))(
                    input_chunk, weight, weight_a, weight_b, r, target_chunk
                )
                grad_weight_a.add_(chunk_grad_weight_a)
                grad_weight_b.add_(chunk_grad_weight_b)
                loss_acc.add_(chunk_loss)
                return chunk_grad_input

            if compiled:
                accumulate_chunk = torch.compile(accumulate_chunk)
            
            input_chunks = torch.chunk(_input, chunks=chunks, dim=0)
            target_chunks = torch.chunk(target, chunks=chunks, dim=0)
            for input_chunk, target_chunk in zip(input_chunks, target_chunks):
                grad_inputs.append(accumulate_chunk(input_chunk, target_chunk))
            
            ctx.save_for_backward(
                torch.cat(grad_inputs, dim=0) / chunks,
                grad_weight_a / chunks,
                grad_weight_b / chunks,
            )
            ctx.is_module = is_module
            ctx.is_deepspeed = is_deepspeed
            if ctx.is_module:
                ctx.m_a = m_a
                ctx.m_b = m_b
            return loss_acc / chunks

    @staticmethod
    def backward(ctx, grad_output):
        (grad_input, grad_weight_a, grad_weight_b) = ctx.saved_tensors
        if ctx.is_module:
            if ctx.is_deepspeed:
                safe_set_full_grad(ctx.m_a.weight, grad_weight_a)
                safe_set_full_grad(ctx.m_b.weight, grad_weight_b)
            else:
                ctx.m_a.weight.grad = grad_weight_a
                ctx.m_b.weight.grad = grad_weight_b

            return (grad_input, None, None, None, None, None, None, None)
        else:
            return (grad_input, None, grad_weight_a, grad_weight_b, None, None, None, None)