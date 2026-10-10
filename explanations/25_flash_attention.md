# Blockwise attention with online softmax

## What the function does

Compute ordinary scaled dot-product attention while visiting small blocks of queries and keys. Full attention builds a score matrix containing every query-key pair. This reference instead keeps only a current score tile and, for each query, a running maximum, denominator, and weighted-value sum.

`Q`, `K`, and `V` all have shape `(B, S, D)`: batch size, sequence length, and feature width. The returned tensor has shape `(B, S, D)`. `block_size` is a positive integer setting the maximum number of query or key positions processed per tile. There are no separate heads or masks in this function.

For each query, the desired result is `sum(exp(score) * value) / sum(exp(score))`, where `score = query dot key / sqrt(D)`. The challenge is combining pieces of these sums safely when a later tile contains a larger score.

## Initialize state for one query block

```python
B, S, D = Q.shape
output = torch.zeros_like(Q)
for i in range(0, S, block_size):
    qi = Q[:, i:i+block_size]
    bs_q = qi.shape[1]
    row_max = torch.full((B, bs_q, 1), float('-inf'), device=Q.device)
    row_sum = torch.zeros(B, bs_q, 1, device=Q.device)
    acc = torch.zeros(B, bs_q, D, device=Q.device)
```

`i` is the first query position of a block. Slicing automatically shortens the final block when `S` is not divisible by `block_size`; `bs_q` records its actual length.

Each query has one `row_max`, initially negative infinity because no score has been seen. `row_sum` holds the sum of exponentials after shifting by the current maximum. `acc` holds the correspondingly shifted weighted sum of values. Both start at zero. These states are reset for each query block because different queries have different scores.

```text
Current queries qi:  (B, bs_q, D)
Current keys kj:     (B, bs_k, D)
Current score tile:  (B, bs_q, bs_k)
Running row maximum: (B, bs_q, 1)
Running denominator: (B, bs_q, 1)
Running numerator:   (B, bs_q, D)
```

The one-entry final axes on the maximum and denominator let them broadcast over key positions or value features.

## Read a key/value tile and update the reference maximum

```python
for j in range(0, S, block_size):
    kj = K[:, j:j+block_size]
    vj = V[:, j:j+block_size]
    scores = torch.bmm(qi, kj.transpose(1, 2)) / math.sqrt(D)
    block_max = scores.max(dim=-1, keepdim=True).values
    new_max = torch.maximum(row_max, block_max)
```

`torch.bmm` performs one matrix multiplication per batch item. The key transpose pairs every current query with every current key. Taking the maximum over the last dimension finds each query's largest score in this tile. `new_max` compares that with the largest score from earlier tiles.

Subtracting this largest score before exponentiation keeps all current exponents at most one. However, previous sums were measured relative to the **old** maximum. They must be converted to the new scale too.

## Rescale the earlier work before adding the new tile

```python
correction = torch.exp(row_max - new_max)
exp_scores = torch.exp(scores - new_max)
acc = acc * correction + torch.bmm(exp_scores, vj)
row_sum = row_sum * correction + exp_scores.sum(dim=-1, keepdim=True)
row_max = new_max
```

If an earlier score is `s` and the maximum changes from `m_old` to `m_new`, then `exp(s - m_new) = exp(s - m_old) * exp(m_old - m_new)`. That second factor is `correction`. Multiplying both prior sums by it expresses all old and new contributions relative to the same maximum.

When the maximum stays unchanged, the factor is one. When it increases, the factor is smaller than one. For the first tile, `exp(-inf - new_max)` is zero, neatly clearing the empty initial state.

Only after visiting every key tile does the code normalize:

```python
output[:, i:i+block_size] = acc / row_sum
return output
```

The common maximum shift cancels between numerator and denominator. Normalizing each tile separately and averaging tile outputs would lose the correct relative probability mass.

## A two-tile calculation you can do by hand

Use width `D = 1`, query `1`, keys `[0, log(2)]`, and values `[2, 10]`. Scores are `[0, log(2)]`; their unshifted exponential weights are `[1, 2]`.

| Step | Running maximum | Correction on old state | New denominator | New numerator |
|---|---|---|---|---|
| Start | `-inf` | — | `0` | `0` |
| First key: score `0`, value `2` | `0` | `0` | `1` | `2` |
| Second key: score `log(2)`, value `10` | `log(2)` | `0.5` | `1*0.5 + 1 = 1.5` | `2*0.5 + 10 = 11` |

The result is `11 / 1.5 = 22 / 3 ≈ 7.3333`, equivalent to weights `[1/3, 2/3]` on `[2, 10]`. If the old numerator were left unscaled, it would incorrectly become `12`.

```python
# Example: Rescale an earlier tile when the next score is larger
Q = torch.ones(1, 2, 1)
K = torch.tensor([[[0.], [math.log(2.)]]])
V = torch.tensor([[[2.], [10.]]])
out = flash_attention(Q, K, V, block_size=1)
assert out.shape == (1, 2, 1)
assert torch.allclose(out, torch.full((1, 2, 1), 22 / 3), atol=1e-6)
old_sum, old_acc = torch.tensor(1.), torch.tensor(2.)
correction = torch.exp(torch.tensor(0.) - torch.tensor(math.log(2.)))
new_sum = old_sum * correction + 1
new_acc = old_acc * correction + 10
assert torch.allclose(new_sum, torch.tensor(1.5))
assert torch.allclose(new_acc, torch.tensor(11.))
assert torch.allclose(new_acc / new_sum, out[0, 0, 0])
```

For a second example, use three keys but tiles of size two. The first tile has scores `[0, log(2)]`, maximum `log(2)`, denominator `1.5`, and numerator `11`. The final one-key tile has score `log(4)` and value `14`. Rescaling by `0.5` gives denominator `1.75` and numerator `19.5`, so the answer is `78 / 7 ≈ 11.1429`.

```python
# Example: Handle a shorter final tile without changing the attention result
Q = torch.ones(1, 3, 1)
K = torch.tensor([[[0.], [math.log(2.)], [math.log(4.)]]])
V = torch.tensor([[[2.], [10.], [14.]]])
out = flash_attention(Q, K, V, block_size=2)
expected = torch.full((1, 3, 1), (1*2 + 2*10 + 4*14) / 7)
assert torch.allclose(out, expected, atol=1e-6)
assert torch.allclose(flash_attention(Q, K, V, block_size=3), expected, atol=1e-6)
```

## Common mistakes and reference scope

Always rescale both the numerator and denominator when updating the maximum. This educational implementation still does quadratic query-key work; its Python loops are not a fused GPU FlashAttention kernel. It assumes matching 3-D shapes and is designed around float32 state; it provides no causal masking, dropout, or general mixed-precision handling.
