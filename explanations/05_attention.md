# Softmax Attention

## Goal and intuition

Attention lets each query gather information from a set of values. A query's similarity to each key determines how much of the corresponding value to include. This exercise implements the core scaled dot-product operation before adding learned projections or multiple heads.

**Shapes:** `Q` is `(B, S_q, D_k)`, `K` is `(B, S_k, D_k)`, and `V` is `(B, S_k, D_v)`. Scores and weights are `(B, S_q, S_k)`; the output is `(B, S_q, D_v)`. Query and key sequence lengths may differ, so the same function supports cross-attention.

## Walk through the reference

1. **Compute similarities.** Transposing `K` from `(B, S_k, D_k)` to `(B, D_k, S_k)` makes `torch.bmm(Q, K.transpose(1, 2))` compare each query with every key in its own batch item.
2. **Scale and normalize the scores.** Division by `sqrt(D_k)` tempers dot-product magnitude. Softmax over the last axis converts the scores for each query into weights over keys, whose sum is one.
3. **Gather values.** `torch.bmm(weights, V)` sums the value vectors using those weights. Each query therefore produces one `D_v`-dimensional output.

## Why it works

The calculation for query position `i` is

$$
a_{ij} = \frac{\exp(Q_i\cdot K_j/\sqrt{D_k})}{\sum_t \exp(Q_i\cdot K_t/\sqrt{D_k})},
\qquad o_i = \sum_j a_{ij}V_j.
$$

Under a rough independent unit-variance feature model, a dot product's variance grows with `D_k`. The square-root scaling keeps the score scale more consistent as feature width changes, reducing premature softmax saturation. Similarity comparisons and weighted sums are differentiable, so training can adjust queries, keys, and values together.

## Pitfalls and reference limits

Softmax must run over keys, not over queries or the batch. Keys and values must agree on sequence length even when queries have a different length. `torch.bmm` requires three-dimensional inputs and does not broadcast mismatched batch sizes. This implementation has no causal mask, padding mask, dropout, or learned projection layers; every query sees every supplied key. It materializes the full score matrix, requiring memory proportional to `B * S_q * S_k`. The judge verifies numerical correctness, autograd, and cross-attention with a different value width.
