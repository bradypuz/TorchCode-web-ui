# GPT-2 Transformer Block

## Goal and intuition

This exercise combines normalization, causal multi-head attention, a feed-forward network, and residual connections into one decoder block. Attention mixes information across allowed sequence positions. The MLP transforms each position's features independently. Residual additions preserve a direct path for information and gradients through both transformations.

**Shapes:** input and output are `(B, S, D)`. With `H` heads and `d_k = D // H`, projected head tensors are `(B, H, S, d_k)`, attention scores are `(B, H, S, S)`, and the MLP's hidden representation is `(B, S, 4D)`. `D` must be divisible by a positive `H`.

## Walk through the reference

1. **Create registered components.** The class inherits from `nn.Module` and calls `super().__init__()`. Two LayerNorms, four linear attention projections, and the sequential MLP are therefore registered for parameter enumeration, device transfer, and serialization.
2. **Compute causal attention in `_attn`.** Q/K/V projections are split into heads. Scaled similarities receive a strict upper-triangular Boolean mask before softmax. The resulting values are concatenated and passed through `W_o`.
3. **Apply the first pre-normalized residual update.** `x + _attn(ln1(x))` normalizes the branch input while the residual path carries the unnormalized current representation.
4. **Apply the second update.** `x + mlp(ln2(x))` operates on the already attention-updated tensor. The MLP expands features to `4D`, applies GELU, and projects back to `D` so the residual addition is shape-compatible.

## Why the architecture stays causal

LayerNorm uses only a token's feature coordinates, and each linear/MLP operation is positionwise. Attention is the only operation that mixes positions, and its mask excludes the future. Consequently, replacing later input tokens cannot change earlier output positions. Pre-norm means normalization happens **before** each sublayer, not after its residual addition; swapping this order gives a different architecture.

## Pitfalls and reference limits

The second residual must use the result of the first update. Preserve model width when merging heads and returning from the MLP. This reference is one simplified GPT-2-style block, not a complete language model: it has no token embeddings, positional embeddings, output vocabulary head, dropout, or KV cache. Head divisibility is assumed rather than checked, and attention uses a dense quadratic score matrix. The judge checks registered components, expansion shapes, causality, output shape, and gradient connectivity for every parameter.
