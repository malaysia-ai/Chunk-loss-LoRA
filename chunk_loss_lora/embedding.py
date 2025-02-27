import torch
import torch.nn as nn

class ChunkedEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _input, weight, weight_a, weight_b, r, compiled=True, CHUNK_SIZE=1024):

        def lookup(input_chunk, weight, weight_a, weight_b, r, target):
            left = input_chunk @ weight.T
            right = (input_chunk @ weight_a.T) @ weight_b.T
            logits = left + right * r
            return logits

        grad_weight_a = torch.zeros_like(weight_a)
        grad_weight_b = torch.zeros_like(weight_b)
        grad_inputs = []
        outputs = []

        B, L = _input.shape

        _input = _input.reshape(-1)
        chunks = _input.shape[0] // CHUNK_SIZE
        def accumulate_chunk(input_chunk):
            (chunk_grad_input, chunk_grad_weight_a, chunk_grad_weight_b), logits = torch.func.grad_and_value(compute_loss, argnums=(0,2,3))(
                input_chunk, weight, weight_a, weight_b, r
            )
            grad_weight_a.add_(chunk_grad_weight_a)
            grad_weight_b.add_(chunk_grad_weight_b)
            return chunk_grad_input, logits

        if compiled:
            accumulate_chunk = torch.compile(accumulate_chunk)
        
        input_chunks = torch.chunk(_input, chunks=chunks, dim=0)
        for input_chunk in input_chunks:
            o = accumulate_chunk(input_chunk)
            grad_inputs.append(o[0])
            outputs.append(o[1])
        
        ctx.save_for_backward(
            torch.cat(grad_inputs, dim=0)/chunks,
            grad_weight_a/chunks,
            grad_weight_b/chunks,
        )
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        (grad_input, grad_weight_A, grad_weight_B) = ctx.saved_tensors
        return (grad_input, None, grad_weight_A, grad_weight_B, None, None, None, None)