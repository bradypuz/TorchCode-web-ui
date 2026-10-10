# Flash Attention (Tiled)

## Goal and intuition

Compute ordinary scaled dot-product attention by visiting small score tiles instead of materializing the entire sequence-by-sequence score matrix. The important challenge is softmax: its normalization spans all key positions, so each tile must update a shared running maximum, denominator, and weighted-value accumulator.

## Inputs and outputs

- `Q`, `K`, `V`: matching floating tensors `(B, S, D)`.
- `block_size`: a positive integer defining query and key tile sizes.
- Output: `(B, S, D)`.
- A score tile: `(B, query_block_length, key_block_length)`.

The final block may be shorter than `block_size`; slicing handles it automatically.

## Walk through the reference

1. Allocate the output, then iterate over query blocks.
2. For each query block, initialize a row maximum to negative infinity, a row sum to zero, and a weighted-value accumulator to zero.
3. Visit each key/value block and compute scores with `bmm(qi, kj.transpose(...)) / sqrt(D)`.
4. Merge the previous row maximum with the current tile's maximum to obtain `new_max`.
5. Rescale previous sums and accumulators by `exp(old_max - new_max)`, and add the current tile's exponentiated scores and weighted values using the new maximum.
6. After every key block has been visited, divide the accumulated weighted values by the accumulated denominator.

## Why it works

For each query row, the maintained quantities satisfy:

$$
m=\max_j s_j,\qquad \ell=\sum_j e^{s_j-m},\qquad a=\sum_j e^{s_j-m}v_j
$$

Changing the running maximum rescales both $\ell$ and $a$ by the same factor. Their final ratio $a/\ell$ is therefore the ordinary softmax-weighted value sum. The first correction factor is zero because the initial maximum is negative infinity. Different tilings agree within floating-point tolerance rather than necessarily bit for bit.

## Pitfalls and reference scope

Rescale **both** the denominator and the accumulator whenever the maximum changes. This is an educational Python/PyTorch implementation of the online-softmax idea, **not a fused CUDA kernel or a guarantee of FlashAttention speed**. Computation remains quadratic; training autograd can retain intermediate tensors. There is no causal mask, padding mask, dropout, or separate value width. The running buffers use the default floating dtype rather than `Q.dtype`; half/bfloat16 inputs can therefore fail at mixed-dtype `bmm`. Block size and input compatibility are not validated. Tests cover ordinary attention agreement, partial tiles, different block sizes, and gradient flow.
