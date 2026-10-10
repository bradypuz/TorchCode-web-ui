# Multi-Head Attention

This class runs several attention calculations in parallel. Each head sees a learned slice of query, key, and value features. The implementation then puts the head outputs back into one vector and learns how to mix them with an output projection. Most of the apparent complexity is bookkeeping: follow the sequence, head, and feature axes carefully.

## Establish the shape vocabulary

`B` is batch size, `S_q` query length, `S_k` key/value length, `D = d_model` model width, and `H = num_heads` head count. Each head has width `d_k = D // H`; `D` must divide evenly by `H`. Inputs are `Q: (B, S_q, D)` and `K/V: (B, S_k, D)`. Output is `(B, S_q, D)`.

For `D=4` and `H=2`, each head receives two feature coordinates. Splitting heads does not create more features: `2 * 2` is still four.

## 1. Create the four projections

```python
self.num_heads = num_heads
self.d_k = d_model // num_heads
self.W_q = nn.Linear(d_model, d_model)
self.W_k = nn.Linear(d_model, d_model)
self.W_v = nn.Linear(d_model, d_model)
self.W_o = nn.Linear(d_model, d_model)
```

The three input projections learn different representations for scoring queries, scoring keys, and carrying values. Their outputs still have width `D`. The default linear-layer biases are included. `W_o` will operate after heads have been joined; it can mix features gathered by different heads.

## 2. Separate the head axis from the sequence axis

```python
B, S_q, _ = Q.shape
S_k = K.shape[1]
q = self.W_q(Q).view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)
k = self.W_k(K).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)
v = self.W_v(V).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)
```

The ignored `_` is the already-known model width. Query and key lengths are read separately because cross-attention can have different numbers of queries and keys.

For a projected value tensor containing two tokens `[1,2,3,4]` and `[5,6,7,8]`, `view` groups each token's features into two adjacent pairs. It changes `(1,2,4)` into `(1,2,2,2)`, interpreted as batch, token, head, head feature. `transpose(1,2)` moves heads before tokens, so matrix multiplication can run independently within each head.

| Head-major value layout | Token 0 | Token 1 |
|---|---|---|
| Head 0 | `[1, 2]` | `[5, 6]` |
| Head 1 | `[3, 4]` | `[7, 8]` |

Neither operation mixes numeric values. It changes which dimensions describe them. Q/K/V all end with layout `(B,H,S,d_k)`.

## 3. Perform one attention calculation per head

```python
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
weights = torch.softmax(scores, dim=-1)
attn = torch.matmul(weights, v)
```

Transposing the last two key axes gives `(B,H,d_k,S_k)`. The product produces `(B,H,S_q,S_k)`. `matmul` handles the batch and head axes as separate leading dimensions. Scale by **head width**, not total model width, because each dot product sums only `d_k` coordinates.

Softmax runs over keys, the last axis. The value multiplication then gives `(B,H,S_q,d_k)`. If a head's two key scores are equal, its weights are `[0.5,0.5]`. In the table, head zero returns `[3,4]` and head one returns `[5,6]`.

## 4. Restore token-major order and concatenate heads

```python
out = attn.transpose(1, 2).contiguous().view(B, S_q, -1)
return self.W_o(out)
```

The transpose makes the layout `(B,S_q,H,d_k)`. Now adjacent head feature groups can be joined into model width, giving `[3,4,5,6]` for our query. `.contiguous()` ensures the transposed tensor's storage layout is compatible with this `.view`; a transpose generally changes strides without rearranging storage. `-1` asks PyTorch to infer `H*d_k = D`.

## Run a head-layout and averaging example

All four projections are explicitly set below, so the result does not depend on random initialization. Zero query/key projections make both heads use equal key weights; identity value/output projections expose the reshape and concatenation directly.

```python
# Example: Two heads average different feature pairs
mha = MultiHeadAttention(d_model=4, num_heads=2)
with torch.no_grad():
    for projection in [mha.W_q, mha.W_k, mha.W_v, mha.W_o]:
        projection.weight.zero_()
        projection.bias.zero_()
    mha.W_v.weight.copy_(torch.eye(4))
    mha.W_o.weight.copy_(torch.eye(4))
Q = torch.zeros(1, 1, 4)
K = torch.zeros(1, 2, 4)
V = torch.tensor([[[1., 2., 3., 4.], [5., 6., 7., 8.]]], requires_grad=True)
heads = mha.W_v(V).view(1, 2, 2, 2).transpose(1, 2)
assert torch.equal(heads[0, 0], torch.tensor([[1., 2.], [5., 6.]]))
assert torch.equal(heads[0, 1], torch.tensor([[3., 4.], [7., 8.]]))
out = mha.forward(Q, K, V)
assert out.shape == (1, 1, 4)
assert torch.allclose(out, torch.tensor([[[3., 4., 5., 6.]]]))
out.sum().backward()
assert torch.allclose(V.grad, torch.full_like(V, 0.5))
assert mha.W_o.weight.grad is not None
```

There is one query and two keys, so this verifies cross-attention as well. Each value coordinate contributes half to the summed output, explaining its gradient.

## Reference scope

The class is a plain Python class: its linear layers support gradients, but the container does not supply `nn.Module` parameter enumeration or device transfer. Head divisibility is assumed. No masks or dropout are applied, and all score matrices are dense.
