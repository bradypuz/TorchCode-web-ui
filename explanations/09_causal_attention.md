# Causal Self-Attention

## Goal and intuition

Autoregressive models predict the next token using only the current prefix. Causal attention enforces that rule inside attention: query position `i` may read keys and values at positions `0` through `i`, including itself, but cannot read later positions.

**Shapes:** `Q` and `K` are `(B, S, D_k)`; `V` is `(B, S, D_v)`. Scores and weights are `(B, S, S)`, the mask is `(S, S)`, and the output is `(B, S, D_v)`. The exercise assumes aligned self-attention sequence positions.

## Walk through the reference

1. **Calculate ordinary attention scores.** `torch.bmm(Q, K.transpose(1, 2)) / sqrt(D_k)` computes scaled query-key similarities for every position pair.
2. **Mark future positions.** `torch.triu(..., diagonal=1)` produces a Boolean upper-triangular mask. `True` entries have column index `j > i`. The diagonal remains unmasked, allowing each position to attend to itself. Creating the mask on `scores.device` makes it usable with either CPU or GPU tensors.
3. **Mask before softmax and gather values.** `masked_fill(..., -inf)` removes future positions from the softmax distribution. Unsqueezing the mask adds a batch axis, letting the same rule broadcast across examples. The final batched multiplication combines only allowed values.

## Why it works

Exponentiating negative infinity yields zero, so masked entries receive zero weight while the remaining entries are renormalized. Each row retains at least its diagonal, ensuring at least one allowed key for nonempty finite inputs. Position zero sees only its own value, so its output is exactly `V[:, 0]` up to numerical precision. Changing later keys or values cannot affect earlier outputs, which is the central property checked by the judge.

## Pitfalls and reference limits

A lower-triangular mask would remove the past instead of the future. Masking after softmax without renormalizing would leave rows with weights summing to less than one. This implementation builds a square mask from the key sequence length; it is intended for equal query and key lengths, not cached decode or general cross-attention. It also has no padding mask, dropout, or projection layers and materializes the complete quadratic score matrix. The supplied tests verify output shape, prefix independence, the first-position property, and gradient connectivity.
