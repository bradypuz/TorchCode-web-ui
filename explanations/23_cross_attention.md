# Multi-Head Cross-Attention

## Goal and intuition

Let one sequence retrieve information from another. Queries come from `x_q`, while keys and values come from `x_kv`. Each query compares itself with every key and uses those comparisons to average the values. Multiple heads perform this retrieval in different learned feature subspaces.

## Inputs and outputs

- `x_q`: `(B, S_q, D)`.
- `x_kv`: `(B, S_kv, D)`; its sequence length can differ from the query length.
- Number of heads: `H`, with `d_k = D / H`.
- Per-head attention scores: `(B, H, S_q, S_kv)`.
- Output: `(B, S_q, D)`, one result for each query position.

Both streams must have the same batch size and model width. The model width must be divisible by the positive head count.

## Walk through the reference

1. Create four learned linear layers: query, key, value, and output projections.
2. Project `x_q` into queries, and `x_kv` separately into keys and values.
3. Reshape each projection into heads and transpose the sequence/head axes, producing `(B, H, S, d_k)`.
4. Multiply queries by transposed keys and divide by `sqrt(d_k)`.
5. Apply softmax over the **key positions**, then multiply those weights by the values.
6. Transpose the heads back, make the tensor contiguous, concatenate their features, and apply `W_o`.

## Why it works

For each head:

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Each query's softmax weights sum to one across the source sequence. The scale moderates dot-product magnitude as head width grows. Concatenating heads restores model width, and the output projection can mix information gathered by different heads.

## Pitfalls and reference scope

Softmax over query positions would normalize the wrong axis. Taking keys or values from `x_q` would turn the intended cross-attention into a different operation. The reference uses **no causal or padding mask**: every key/value position can influence every query, including any padded positions supplied by the caller. It has no attention dropout, residual connection, or normalization. Constructor arguments are not validated explicitly, so incompatible head dimensions fail during division or reshaping. The judge checks different sequence lengths, visibility of the final source position, output shape, and gradients to both input streams.
