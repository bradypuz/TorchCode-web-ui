# Multi-head cross-attention

## What the layer does

For each vector in one sequence, read useful information from another sequence. The first sequence supplies **queries**: what to look for. The second supplies **keys** used for matching and **values** containing what to return. Their lengths can differ, which is why the code tracks both separately.

`x_q` has shape `(B, S_q, d_model)` and `x_kv` has shape `(B, S_kv, d_model)`. `B` is the shared batch size, `S_q` the number of query positions, and `S_kv` the number of source positions. With `H = num_heads`, each head has width `d_k = d_model // H`; `d_model` must be divisible by `H`. The output has shape `(B, S_q, d_model)`: one result per query, regardless of the source length.

## Create the projections and split into heads

```python
self.num_heads = num_heads
self.d_k = d_model // num_heads
self.W_q = nn.Linear(d_model, d_model)
self.W_k = nn.Linear(d_model, d_model)
self.W_v = nn.Linear(d_model, d_model)
self.W_o = nn.Linear(d_model, d_model)
```

The constructor first calls `super().__init__()` so these layers are registered. Each projection has its own weights and default bias. `W_q`, `W_k`, and `W_v` transform the input into matching and content features. `W_o` combines the heads' results afterward.

```python
B, S_q, _ = x_q.shape
S_kv = x_kv.shape[1]
q = self.W_q(x_q).view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)
k = self.W_k(x_kv).view(B, S_kv, self.num_heads, self.d_k).transpose(1, 2)
v = self.W_v(x_kv).view(B, S_kv, self.num_heads, self.d_k).transpose(1, 2)
```

`.view(...)` divides the final feature axis into heads. `.transpose(1, 2)` then puts the head axis before the sequence axis, so matrix multiplication can operate independently on every batch/head pair.

```text
queries:       (B, S_q,  d_model) → (B, H, S_q,  d_k)
keys/values:   (B, S_kv, d_model) → (B, H, S_kv, d_k)
scores:                              (B, H, S_q, S_kv)
head results:                        (B, H, S_q, d_k)
```

Notice that keys and values are projected from `x_kv`, not `x_q`. Accidentally using the query sequence for all three would make this self-attention instead.

## Match queries and average source values

```python
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
weights = torch.softmax(scores, dim=-1)
attn = torch.matmul(weights, v)
```

Transposing the key's last two axes permits a dot product between every query and every source key. Dividing by `sqrt(d_k)` moderates score size as the head width increases. Softmax on the last axis turns each query's source scores into nonnegative weights summing to one.

Multiplying those weights by `v` forms a weighted sum of source values. Different queries can use different mixtures, and heads can use different matching features. This reference includes no causal or padding mask, so every query can read every source position.

## Reassemble the heads

```python
return self.W_o(attn.transpose(1, 2).contiguous().view(B, S_q, -1))
```

The transpose restores `(B, S_q, H, d_k)`. A transpose often leaves a noncontiguous view of memory; `.contiguous()` arranges its values so `.view(...)` can safely join the final two axes. `-1` asks PyTorch to infer their combined width, `d_model`. The output projection then mixes those joined features.

## A concrete query reading two source vectors

Set all projections to the identity, with zero biases, and use one head of width two. Source vectors are `[1, 0]` and `[0, 2]`. A query `[sqrt(2), 0]` has unscaled dot products `[sqrt(2), 0]` and therefore scaled scores `[1, 0]`.

| Query | Scaled scores | Softmax weights | Weighted source result |
|---|---|---|---|
| `[sqrt(2), 0]` | `[1, 0]` | `[0.7311, 0.2689]` | `[0.7311, 0.5379]` |
| `[0, 0]` | `[0, 0]` | `[0.5, 0.5]` | `[0.5, 1]` |

The zero query gives equal scores and reads the arithmetic average. The first query favors the first source without completely ignoring the second.

```python
# Example: Two queries read weighted mixtures of two source vectors
layer = MultiHeadCrossAttention(d_model=2, num_heads=1)
with torch.no_grad():
    for projection in (layer.W_q, layer.W_k, layer.W_v, layer.W_o):
        projection.weight.copy_(torch.eye(2))
        projection.bias.zero_()
x_q = torch.tensor([[[math.sqrt(2.), 0.], [0., 0.]]])
x_kv = torch.tensor([[[1., 0.], [0., 2.]]])
out = layer(x_q, x_kv)
s = torch.sigmoid(torch.tensor(1.)).item()
expected = torch.tensor([[[s, 2 * (1 - s)], [0.5, 1.]]])
assert out.shape == (1, 2, 2)
assert torch.allclose(out, expected, atol=1e-6)
assert layer(x_q[:, :1], x_kv).shape == (1, 1, 2)
```

The final assertion changes the query length to one while keeping two source positions, confirming that the result length follows queries.

## Common mistakes and reference scope

Normalize over source positions, not queries or features. Use the correct sequence length when splitting keys and values into heads, and preserve the contiguous step before `.view`. This implementation assumes compatible batches and widths and provides no masks, dropout, cached keys/values, or explicit divisibility check.
