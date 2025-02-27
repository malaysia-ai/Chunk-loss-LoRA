import torch
import torch.nn as nn

ce = nn.CrossEntropyLoss()

class ChunkedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _input, weight, weight_a, weight_b, r, target, compiled=True, CHUNK_SIZE=1024):

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

        chunks = _input.shape[0] // CHUNK_SIZE
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
            torch.cat(grad_inputs, dim=0)/chunks,
            grad_weight_a/chunks,
            grad_weight_b/chunks,
        )
        return loss_acc / chunks

    @staticmethod
    def backward(ctx, grad_output):
        (grad_input, grad_weight_A, grad_weight_B) = ctx.saved_tensors
        return (grad_input, None, grad_weight_A, grad_weight_B, None, None, None, None)