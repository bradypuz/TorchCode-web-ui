# Softmax Attention

Attention answers a query by blending available value vectors. A key describes when its matching value should be useful. The reference has no learned projections yet: you supply `Q`, `K`, and `V` directly, and three statements compute similarities, normalize them, and use them to gather values.

## Give every dimension a meaning

`B` is batch size, `S_q` is query count, `S_k` is key/value count, `D_k` is query/key width, and `D_v` is value width.

| Tensor | Shape | Meaning |
|---|---|---|
| `Q` | `(B, S_q, D_k)` | One question per query position |
| `K` | `(B, S_k, D_k)` | Descriptions to compare with queries |
| `V` | `(B, S_k, D_v)` | Information attached to the keys |
| Scores and weights | `(B, S_q, S_k)` | One key distribution per query |
| Output | `(B, S_q, D_v)` | One blended value per query |

`S_q` and `S_k` may differ. For example, three decoder queries can attend to five encoder values. Only keys and values must share their sequence length, because key `j` supplies the weight for value `j`.

## 1. Compare every query with every key

```python
d_k = K.size(-1)
scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
```

`K.size(-1)` reads the feature width used in each dot product. Transposing dimensions one and two changes keys from `(B, S_k, D_k)` to `(B, D_k, S_k)`. That puts `D_k` in the position required for matrix multiplication: `(S_q, D_k) @ (D_k, S_k)` produces `(S_q, S_k)` for each batch item.

`torch.bmm` performs that multiplication separately across the batch; it requires three-dimensional tensors. Score `(b, i, j)` is the dot product of query `i` and key `j`, divided by `sqrt(D_k)`. Wider dot products tend to have larger magnitudes, so the scale helps keep softmax from becoming excessively sharp merely because width increased.

## 2. Normalize over keys, not queries

```python
weights = torch.softmax(scores, dim=-1)
```

The last score axis is `S_k`. Softmax turns each query's row of key scores into non-negative weights that sum to one. Each query has its own row; one query's weights do not compete with another query's weights.

Suppose `D_k=2`, keys are `[1, 0]` and `[0, 1]`, and a query is `[sqrt(2)*log(3), 0]`. Its scaled scores are `[log(3), 0]`. Exponentiation gives `[3, 1]`, and normalization gives `[0.75, 0.25]`.

## 3. Combine the corresponding values

```python
return torch.bmm(weights, V)
```

The shape calculation is `(B, S_q, S_k) @ (B, S_k, D_v) -> (B, S_q, D_v)`. This multiplication sums across keys. For scalar values `10` and `20`, the example query returns `0.75*10 + 0.25*20 = 12.5`. A query with scores `[0, 0]` gives weights `[0.5, 0.5]` and returns `15`.

```text
query → dot products with keys → divide by sqrt(D_k)
      → scores [log(3), 0]
      → softmax weights [0.75, 0.25]
      → 0.75 × value 10 + 0.25 × value 20
      → output 12.5
```

```torchcode-visual
attention
```

## Run both equal and unequal weighting

```python
# Example: One query chooses between two scalar values
K = torch.tensor([[[1., 0.], [0., 1.]]], requires_grad=True)
V = torch.tensor([[[10.], [20.]]], requires_grad=True)
Q = torch.tensor([[[math.sqrt(2.) * math.log(3.), 0.]]], requires_grad=True)
scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(2.)
weights = torch.softmax(scores, dim=-1)
assert scores.shape == (1, 1, 2)
assert torch.allclose(weights, torch.tensor([[[0.75, 0.25]]]))
out = scaled_dot_product_attention(Q, K, V)
assert out.shape == (1, 1, 1)
assert torch.allclose(out, torch.tensor([[[12.5]]]))
uniform = scaled_dot_product_attention(torch.zeros(1, 1, 2), K, V)
assert torch.allclose(uniform, torch.tensor([[[15.]]]))
out.sum().backward()
assert torch.allclose(V.grad, torch.tensor([[[0.75], [0.25]]]))
assert Q.grad is not None and K.grad is not None
```

The query count is one while the key/value count is two, so this also exercises cross-attention. The value gradient equals its attention weight for this scalar summed loss: raising the first value by one raises the output by `0.75`.

## What this reference includes

All operations support autograd, letting a surrounding model learn the supplied representations. This function has no causal mask, padding mask, dropout, or projection parameters: every query can use every key. It materializes the full `(B, S_q, S_k)` score matrix, so memory grows with the product of the two sequence lengths. Matching batch sizes, feature widths, and key/value counts are assumed.
