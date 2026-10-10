# Sliding Window Attention

## Goal and intuition

Sliding-window attention restricts each query to nearby sequence positions. This reference uses a **symmetric** neighborhood: position `i` may attend to positions from `i - window_size` through `i + window_size`, clipped by sequence boundaries. It can therefore include future neighbors; it is not causal attention.

**Shapes:** `Q` and `K` are `(B, S, D_k)`, `V` is `(B, S, D_v)`, and the output is `(B, S, D_v)`. Pairwise scores and the distance mask are `(B, S, S)` and `(S, S)` respectively. The intended window size is a non-negative integer radius.

## Walk through the reference

1. **Calculate full attention scores.** A batched query-key matrix product, scaled by `sqrt(D_k)`, initially compares every position with every other position.
2. **Construct the distance rule.** `arange(S)` lists token indices. Broadcasting one row-shaped and one column-shaped copy creates all pairwise index differences. Taking the absolute value gives distances `|i-j|`; positions with distance greater than `window_size` are marked for masking.
3. **Exclude distant keys and combine values.** The mask broadcasts across the batch. Replacing excluded scores with negative infinity before softmax gives them zero probability. The weighted value product produces one output vector for each query.

## Why it works

Every allowed row includes its own diagonal, so a non-negative radius leaves at least one finite score for ordinary inputs. Two boundary cases make the contract especially clear. With `window_size=0`, the only allowed key is the same position, giving output `V` regardless of its query-key similarity. With a radius covering the entire sequence, no positions are masked, so the result equals full unmasked attention. Interior positions have at most `2 * window_size + 1` allowed keys, while edge positions have fewer.

## Pitfalls and reference limits

The absolute difference matters: a signed comparison would incorrectly remove only one side of the neighborhood. A negative radius masks every position and yields undefined all-masked softmax rows; the reference does not validate this input. Most importantly, applying a band mask after computing dense scores does **not** make this implementation computationally sparse. It still allocates quadratic score and mask matrices and performs dense matrix multiplications. The exercise demonstrates local attention semantics, rather than a fast sparse kernel. Tests check shape, zero and large radii, independence from distant values, and gradients.
