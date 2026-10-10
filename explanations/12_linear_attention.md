# Linear Self-Attention

## Goal and intuition

This attention variant replaces softmax similarities with a positive feature-map kernel. Its key optimization is matrix multiplication order: summarize keys and values first, then query that summary. It avoids constructing a sequence-by-sequence attention matrix, making work linear in sequence length when feature widths are fixed.

**Shapes:** `Q` and `K` are `(B, S, D_k)`, `V` is `(B, S, D_v)`, and the output is `(B, S, D_v)`. The key-value summary is `(B, D_k, D_v)`; the key sum is `(B, 1, D_k)`; the normalization denominator is `(B, S, 1)`.

## Walk through the reference

1. **Apply the kernel feature map.** `F.elu(tensor) + 1` gives linear positive features for non-negative inputs and exponentially small positive features for negative inputs in exact arithmetic. Both queries and keys use this same map.
2. **Build a compact summary.** `K_prime.transpose(1, 2) @ V` accumulates key-feature/value products across positions. Separately, summing `K_prime` along the sequence produces the key-feature totals needed to normalize attention.
3. **Read and normalize the summary.** Multiplying `Q_prime` by the key-value summary gives the numerator. Multiplying it by the transposed key sum gives one denominator per query. Division by `den + 1e-6` broadcasts over value features and protects against very small denominators.

## Why changing multiplication order works

For query `i`, the intended kernel weights are proportional to `phi(Q_i) · phi(K_j)`:

$$
o_i = \frac{\sum_j (\phi(Q_i)^T\phi(K_j))V_j}{\sum_j \phi(Q_i)^T\phi(K_j)+\epsilon}.
$$

Associativity moves the sequence reduction into `phi(K)^T V`, whose dimensions depend on feature widths rather than `S²`. The main computation costs approximately `O(B * S * D_k * D_v)`. This is a different attention rule from softmax, not an exact shortcut for the softmax result.

## Pitfalls and reference limits

Building `Q_prime @ K_prime.T` first would lose the linear-complexity benefit. The global summary includes every key, so this reference is noncausal and has no padding mask; causal linear attention would require prefix summaries. Finite-precision ELU-plus-one can round very negative features to zero, and epsilon slightly changes normalized weights. No special accumulation dtype is used. The judge checks shape, finite random outputs, gradients, and runtime on long sequences, but does not numerically compare the result with softmax attention.
