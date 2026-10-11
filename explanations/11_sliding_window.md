This function computes **sliding-window attention**: each position attends only to nearby positions, rather than the entire sequence.

It uses the usual scaled dot-product attention calculation, then masks out positions that are too far away.

## 1. Expected input shapes

The function expects three-dimensional tensors:

| Input | Shape | Meaning |
|---|---|---|
| `Q` | `(B, S, d_k)` | Queries: what each position is looking for |
| `K` | `(B, S, d_k)` | Keys: what each position can be matched against |
| `V` | `(B, S, d_v)` | Values: the information each position contributes |

Here:
- `B` is the batch size.
- `S` is the sequence length.
- `d_k` is the query/key dimension.
- `d_v` is the value dimension.

The output has shape **`(B, S, d_v)`**.

## 2. Compute attention scores

```python
d_k = K.size(-1)
scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
```

`K.transpose(1, 2)` changes the key tensor’s shape from:

```text
(B, S, d_k) → (B, d_k, S)
```

`torch.bmm` performs matrix multiplication separately for each item in the batch:

```text
(B, S, d_k) × (B, d_k, S) → (B, S, S)
```

So:

```python
scores[b, i, j]
```

measures how well the query at position `i` matches the key at position `j` in batch item `b`.

Dividing by `sqrt(d_k)` keeps dot products from growing too large as the feature dimension increases, which helps avoid an excessively sharp softmax.

## 3. Build the sliding-window mask

```python
S = Q.size(1)
idx = torch.arange(S, device=Q.device)
```

This creates position indices:

```text
[0, 1, 2, ..., S - 1]
```

Then:

```python
mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > window_size
```

Broadcasting produces an `(S, S)` matrix of distances between positions. The mask is `True` wherever:

```text
|query_position - key_position| > window_size
```

Those positions are **not allowed** to attend to one another.

For example, with `S = 5` and `window_size = 1`, the allowed attention pattern is:

```text
                Key position
                0  1  2  3  4
Query position
       0        ✓  ✓  ·  ·  ·
       1        ✓  ✓  ✓  ·  ·
       2        ·  ✓  ✓  ✓  ·
       3        ·  ·  ✓  ✓  ✓
       4        ·  ·  ·  ✓  ✓
```

**`window_size` is a radius**, not the total window width. An interior position can attend to up to `2 * window_size + 1` positions, including itself.

## 4. Exclude positions outside the window

```python
scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
```

`mask.unsqueeze(0)` has shape `(1, S, S)`, so the same mask is broadcast across every batch item.

Disallowed scores are replaced with negative infinity.

## 5. Convert scores into attention weights

```python
weights = torch.softmax(scores, dim=-1)
```

Softmax runs over the key positions for each query.

- Masked positions receive weight `0`, because `exp(-inf) = 0`.
- Allowed positions receive nonnegative weights that sum to `1`.

Thus, each query distributes its attention only among positions inside its window.

## 6. Combine the values

```python
return torch.bmm(weights, V)
```

This multiplies:

```text
(B, S, S) × (B, S, d_v) → (B, S, d_v)
```

For each query position, the output is a weighted sum of the nearby value vectors:

\[
\text{output}_{b,i}
=
\sum_{j:\,|i-j|\leq \text{window\_size}}
\text{weights}_{b,i,j}\,V_{b,j}.
\]

## Important details

- **This is not causal attention.** A position can attend to both earlier and later positions within its window.
- **It still computes the full attention matrix.** Although distant positions are masked, the implementation allocates `(B, S, S)` scores and performs dense attention computations. It therefore does not obtain the computational savings of a specialized sliding-window implementation.
- **It assumes matching sequence lengths** for `Q`, `K`, and `V`, and it does not mask padding tokens.