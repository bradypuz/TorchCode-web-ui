# Grouped Query Attention

## Goal and intuition

Grouped query attention keeps many query heads while sharing fewer key/value heads across them. Different queries can ask different questions of the same key/value representation. This reduces the width of K/V projections and is the basis for reducing cache storage in a generation implementation.

**Shapes:** input and output are `(B, S, D)`. With `H` query heads, `H_kv` key/value heads, and `d_k = D // H`, queries are `(B, H, S, d_k)` while unexpanded keys and values are `(B, H_kv, S, d_k)`. The reference requires `D % H == 0` and `H % H_kv == 0`.

## Walk through the reference

1. **Project queries at full width and K/V at reduced width.** `W_q` emits `D` features; `W_k` and `W_v` each emit only `H_kv * d_k`. The projected tensors are reshaped and transposed into head-major order.
2. **Expand shared heads.** `repeat_interleave(H // H_kv, dim=1)` copies each K/V head for a consecutive group of query heads. For `H=4` and `H_kv=2`, the mapping is `[kv0, kv0, kv1, kv1]`, rather than alternating groups.
3. **Compute attention and mix outputs.** Scaled dot products and softmax run separately for each query head against its assigned shared keys. Weighted values are concatenated back to model width, then transformed by `W_o`.

## Why it works

All query heads remain distinct because `W_q` has separate output coordinates. Heads in one group share the same projected K/V, but their different queries can still produce different attention distributions. When `H_kv == H`, each group has one member and the structure becomes ordinary multi-head self-attention. When `H_kv == 1`, every query head shares a single K/V head.

## Pitfalls and reference limits

`repeat` and `repeat_interleave` assign groups differently; the contiguous grouping here is part of the exercise. Invalid head counts are not validated and fail through division or shape mismatches. Although projection widths are smaller, this implementation explicitly copies expanded K/V tensors and creates all `H` dense attention matrices. It neither maintains a cache nor implements a memory-efficient grouped kernel. The enclosing class is also a plain Python class rather than `nn.Module`, so its linear layers need explicit parameter/device management. There is no causal mask. Tests inspect reduced projection shapes, sharing, the full-head special case, and autograd.
