# Multi-Head Attention

## Goal and intuition

Multi-head attention learns several attention patterns in parallel. Each head uses its own projected features to score keys and combine values, then an output projection mixes the resulting head representations. Splitting into heads changes the attention structure while preserving the overall model width.

**Shapes:** `Q` is `(B, S_q, D)`, while `K` and `V` are `(B, S_k, D)`. With `H` heads, the reference sets `d_k = D // H`, requiring `D` to be divisible by `H`. Head tensors are `(B, H, S, d_k)`, scores are `(B, H, S_q, S_k)`, and the output is `(B, S_q, D)`.

## Walk through the reference

1. **Learn Q/K/V representations.** Four `nn.Linear(D, D)` layers provide query, key, value, and output projections. Their default biases are included. Each initial projection is reshaped from `(B, S, D)` to `(B, S, H, d_k)` and transposed into head-major order.
2. **Attend independently inside each head.** `torch.matmul` compares each head's queries and keys over the last two dimensions. Division by `sqrt(d_k)` scales the scores, softmax normalizes over key positions, and a second matrix product gathers values.
3. **Concatenate and mix heads.** Transposing the attention result back gives `(B, S_q, H, d_k)`. `.contiguous()` makes its storage compatible with the following `.view(B, S_q, -1)`. `W_o` maps the concatenated features to the final model-width output.

## Why it works

Each head computes scaled dot-product attention in a different learned feature subspace. The projection weights can therefore represent different relationships even though all heads receive the same source tensors. Cross-attention works because the query reshape uses `S_q`, while both key and value reshapes use `S_k`. Only their feature widths must match the model width.

## Pitfalls and reference limits

Using one sequence length for all three reshapes breaks cross-attention. Forgetting `.contiguous()` after the transpose can make `.view()` fail. The constructor does not validate head divisibility, so invalid widths fail later during reshaping. This reference class is not an `nn.Module`: its individual linear layers support autograd, but the enclosing class has no automatic parameter enumeration, device transfer, or serialization. It also provides no mask or dropout, so it is unmasked attention. The tests cover projections, head diversity, cross-attention, numerical results, and gradient flow.
