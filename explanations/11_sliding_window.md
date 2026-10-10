# Sliding Window Attention

This attention function permits a query to read only nearby keys. The window is symmetric: query `i` can use positions from `i-window_size` through `i+window_size`, clipped at the sequence edges. Understanding the index-difference matrix explains almost the entire mask construction.

## Establish the shapes and neighborhood rule

`B` is batch size, `S` sequence length, `D_k` query/key width, and `D_v` value width. Q/K are `(B,S,D_k)`, V is `(B,S,D_v)`, and output is `(B,S,D_v)`. Scores are `(B,S,S)`, with query index on rows and key index on columns.

`window_size` is a non-negative radius. A radius of one allows the previous position, the current position, and the next position. Near an edge, one of those neighbors does not exist, so fewer keys are allowed.

## 1. Compute all scaled query/key scores

```python
d_k = K.size(-1)
scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
S = Q.size(1)
```

The key transpose aligns its feature dimension for batched matrix multiplication. Each score is a query/key dot product divided by the square root of its feature width. The sequence length is read from Q to construct a same-length neighborhood mask.

This reference computes the entire score matrix first. Its mask changes which entries contribute to the answer; it does not prevent those dot products from being calculated.

## 2. Build all distances through broadcasting

```python
idx = torch.arange(S, device=Q.device)
mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > window_size
```

For `S=4`, `idx` is `[0,1,2,3]`. `unsqueeze(0)` makes a row shaped `(1,4)`; `unsqueeze(1)` makes a column shaped `(4,1)`. Broadcasting subtracts every column's query index from every row's key index, giving `j-i`. Taking `.abs()` makes past and future distances equivalent.

```text
absolute distances             mask for window_size=1
          keys 0 1 2 3                     keys 0 1 2 3
query 0        0 1 2 3           query 0        . . X X
query 1        1 0 1 2           query 1        . . . X
query 2        2 1 0 1           query 2        X . . .
query 3        3 2 1 0           query 3        X X . .

. = allowed     X = distance greater than the radius
```

The strict `>` comparison keeps distances exactly equal to the radius. The diagonal distance is zero, so every query keeps at least itself for a non-negative window.

## 3. Exclude distant keys, normalize, and gather values

```python
scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
weights = torch.softmax(scores, dim=-1)
return torch.bmm(weights, V)
```

The `(1,S,S)` mask broadcasts across batches. Forbidden scores become negative infinity and receive zero softmax weight. Normalizing on the last axis distributes weight among a query's **allowed keys**. The final product sums the corresponding values.

With all allowed scores equal and values `[2,4,8,16]`, radius one gives `[3,14/3,28/3,12]`: edge queries average two values; interior queries average three. In particular, query one uses values `2`, `4`, and `8`, not all four values.

```torchcode-visual
attention-mask
```

## Run three useful window sizes

```python
# Example: Local averages expand as the radius increases
Q = torch.zeros(1, 4, 2)
K = torch.zeros(1, 4, 2)
V = torch.tensor([[[2.], [4.], [8.], [16.]]])
idx = torch.arange(4)
mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > 1
assert torch.equal(mask, torch.tensor([[False, False, True, True],
                                     [False, False, False, True],
                                     [True, False, False, False],
                                     [True, True, False, False]]))
local = sliding_window_attention(Q, K, V, window_size=1)
assert local.shape == (1, 4, 1)
assert torch.allclose(local, torch.tensor([[[3.], [14. / 3.], [28. / 3.], [12.]]]))
assert torch.equal(sliding_window_attention(Q, K, V, window_size=0), V)
assert torch.allclose(sliding_window_attention(Q, K, V, window_size=4), torch.full_like(V, 7.5))
changed = V.clone()
changed[:, 3] = 160.
assert torch.allclose(sliding_window_attention(Q, K, changed, 1)[:, 0], local[:, 0])
```

A zero radius leaves only the diagonal, so each output is exactly its own value. A radius covering the sequence becomes full attention, whose uniform result is the global average `7.5`. Changing the distant final value cannot affect the first query's radius-one answer.

## Gradients and reference scope

The attention arithmetic remains differentiable through permitted scores and values. This is a symmetric neighborhood, so it includes future neighbors and is not a causal decoder mask. A negative radius would mask every key and lead to undefined softmax rows. The reference assumes equal sequence lengths, has no padding mask, and still allocates and computes dense quadratic matrices. It demonstrates the local-attention rule rather than an optimized sparse implementation.
