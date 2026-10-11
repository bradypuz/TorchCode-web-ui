This function implements **causal scaled dot-product attention**: each position combines information from **itself and earlier positions**, but cannot use information from later positions. This is the attention pattern used in autoregressive models, where future tokens must remain hidden.

## 1. Inputs and shapes

The function expects three batched tensors:

| Tensor | Meaning | Shape |
|---|---|---|
| `Q` — queries | What each position is looking for | `(B, S, d_k)` |
| `K` — keys | What each position can be matched against | `(B, S, d_k)` |
| `V` — values | The information each position contributes | `(B, S, d_v)` |

Here:
- `B` is the batch size.
- `S` is the sequence length.
- `d_k` is the query/key vector dimension.
- `d_v` is the value vector dimension; it can differ from `d_k`.

Usually, `Q`, `K`, and `V` come from learned transformations of the input embeddings. This function assumes they have already been computed.

## 2. Compute attention scores

```python
d_k = K.size(-1)
scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
```

`K.size(-1)` gets the key vector dimension.

`K.transpose(1, 2)` changes its shape from `(B, S, d_k)` to `(B, d_k, S)`. Then `torch.bmm` performs matrix multiplication separately for each batch item:

```text
(B, S, d_k) × (B, d_k, S) → (B, S, S)
```

Each score measures how well a query at position `i` matches a key at position `j`:

\[
\text{scores}_{i,j}=\frac{Q_i\cdot K_j}{\sqrt{d_k}}
\]

**Rows represent query positions; columns represent key positions.**

Dividing by \(\sqrt{d_k}\) keeps dot products from growing too large as the vector dimension increases. This helps prevent softmax from becoming excessively peaked.

## 3. Create a mask for future positions

```python
S = scores.size(-1)
mask = torch.triu(
    torch.ones(S, S, device=scores.device, dtype=torch.bool),
    diagonal=1
)
```

This creates a Boolean matrix whose entries are `True` **above the main diagonal**.

For a sequence of length 4:

```text
         Key position
          0      1      2      3
Query 0  False  True   True   True
      1  False  False  True   True
      2  False  False  False  True
      3  False  False  False  False
```

`True` means “block this connection.”

- Position 0 can attend only to position 0.
- Position 1 can attend to positions 0 and 1.
- Position 2 can attend to positions 0, 1, and 2.
- And so on.

`diagonal=1` leaves the main diagonal unmasked, so each position can attend to itself.

## 4. Set forbidden scores to negative infinity

```python
scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
```

`mask.unsqueeze(0)` changes the mask shape from `(S, S)` to `(1, S, S)`, allowing the same mask to broadcast across all batch items.

Every future-position score becomes `-inf`. In the next softmax operation, those entries receive zero weight because:

\[
e^{-\infty}=0
\]

## 5. Convert scores into attention weights

```python
weights = torch.softmax(scores, dim=-1)
```

Softmax runs across the key positions for each query.

For each query position:
- Allowed positions receive nonnegative weights that sum to 1.
- Future positions receive weight 0.

For example, position 2 might get:

```text
[0.2, 0.3, 0.5, 0.0]
```

It assigns different importance to positions 0–2 and ignores position 3.

## 6. Combine the value vectors

```python
return torch.bmm(weights, V)
```

The attention weights are used to compute a weighted sum of the value vectors:

\[
\text{output}_i=\sum_{j=0}^{i}\text{weights}_{i,j}V_j
\]

The output shape is:

```text
(B, S, S) × (B, S, d_v) → (B, S, d_v)
```

So, using the example weights above:

```python
output_at_position_2 = 0.2 * V[0] + 0.3 * V[1] + 0.5 * V[2]
```

Here, the indexing illustrates positions within one batch item.

**In short:** `Q` and `K` decide **which earlier positions matter**, the causal mask prevents looking into the future, and `V` provides the information to combine. This function computes one batched attention operation; it does not include multi-head splitting, padding masks, or dropout.