# KV Cache Attention

## Goal and intuition

During autoregressive generation, earlier tokens' projected keys and values do not change. A KV cache reuses those tensors instead of projecting the entire prefix on every step. Only the new tokens need Q/K/V projection, although each new query still attends across the available prefix.

**Shapes:** `x` is `(B, S_new, D)`. With `H` heads and `d_k = D // H`, the optional cache contains two tensors `(B, H, S_past, d_k)`. The returned output is `(B, S_new, D)`, and the new cache contains `(B, H, S_past + S_new, d_k)` keys and values.

## Walk through the reference

1. **Project just the supplied tokens.** The four registered linear layers provide the usual attention projections. New Q/K/V tensors are reshaped into `(B, H, S_new, d_k)`.
2. **Append cached state.** If a cache exists, `torch.cat(..., dim=2)` joins past and new keys and values along their sequence dimension. Queries remain limited to new tokens. The assembled K/V tensors become the returned cache.
3. **Apply the correct causal offset.** For multiple new tokens, the mask is `(S_new, S_total)` with upper-triangular diagonal `S_past + 1`. New query `i` can see past keys and new keys through absolute position `S_past + i`. A single new query needs no mask because every assembled key is already at or before its position.
4. **Attend and return state.** Scaled scores, softmax, and the value product produce new-token representations. Heads are merged, `W_o` projects the result, and the function returns both output and cache.

## Why cached decoding matches full attention

In a full causal pass, a token's result depends only on its prefix. Caching preserves exactly the K/V projections for that prefix, so processing later tokens incrementally presents the same allowed keys and values to each query. The offset mask also supports appending a chunk of several new tokens without letting its earlier queries see later ones.

## Pitfalls and reference limits

Concatenating along the head or feature axis corrupts cache structure. Reusing a cache from another sequence or changed model weights is invalid. The reference assumes compatible shapes, device, dtype, and divisible head counts; it does not validate them. Cache tensors remain attached to autograd, and repeated concatenation copies growing prefixes. Use an inference context when appropriate; this is not a preallocated or detached production cache. The judge verifies cache growth, structure, full-versus-incremental equality, output shape, and gradients.
