This function implements **scaled dot-product attention**, a core operation in Transformers. It lets each query select relevant information from a set of values by comparing the query with corresponding keys.

The formula is:

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

## 1. What are `Q`, `K`, and `V`?

They are tensors representing:

- **Queries (`Q`)**: what each position is looking for.
- **Keys (`K`)**: what each available position can be matched against.
- **Values (`V`)**: the information retrieved from those positions.

Think of a query as a search request, keys as descriptions used to find matches, and values as the actual content you retrieve.

Typically, these tensors are produced by learned linear transformations of token representations. This function assumes they have already been calculated.

### Expected shapes

Because the code uses `torch.bmm`—**batched matrix multiplication**—the inputs must be 3-dimensional:

| Tensor | Shape | Meaning |
|---|---|---|
| `Q` | `(B, L_q, d_k)` | A batch of query vectors |
| `K` | `(B, L_k, d_k)` | A batch of key vectors |
| `V` | `(B, L_k, d_v)` | A batch of value vectors |

Here, `B` is the batch size, `L_q` is the number of queries, and `L_k` is the number of key/value pairs. Queries and keys must have the same feature dimension, `d_k`; values can have a different dimension, `d_v`.

## 2. Line-by-line explanation

### Get the key dimension

```python
d_k = K.size(-1)
```

`size(-1)` returns the size of the last dimension of `K`: the number of features in each key vector.

For example, if `K.shape` is `(2, 5, 64)`, then `d_k = 64`.

### Calculate and scale the attention scores

```python
scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
```

First, `K.transpose(1, 2)` swaps the sequence and feature dimensions:

```text
K:                 (B, L_k, d_k)
K.transpose(1, 2): (B, d_k, L_k)
```

Then `torch.bmm` multiplies each batch’s query matrix by its transposed key matrix:

```text
(B, L_q, d_k) × (B, d_k, L_k) → (B, L_q, L_k)
```

Each entry in `scores` is a dot product between one query and one key. A larger score means that key will receive more attention from that query.

Dividing by \(\sqrt{d_k}\) keeps score magnitudes under control. Without scaling, larger feature dimensions can produce large dot products, making softmax overly concentrated and reducing useful gradients.

### Turn scores into attention weights

```python
weights = torch.softmax(scores, dim=-1)
```

Softmax runs across the **keys** for each query.

For every query, it produces nonnegative weights that sum to 1. For example:

```text
[0.1, 0.7, 0.2]
```

This means the query assigns 10% of its attention to the first key, 70% to the second, and 20% to the third.

The shape remains `(B, L_q, L_k)`.

### Combine the values

```python
return torch.bmm(weights, V)
```

This computes a weighted sum of the value vectors for each query.

With the weights above, one query’s output would be:

\[
0.1V_1 + 0.7V_2 + 0.2V_3
\]

The output shape is:

```text
(B, L_q, L_k) × (B, L_k, d_v) → (B, L_q, d_v)
```

So the function returns **one attention-weighted value vector per query**, not the attention weights themselves.

## In short

**Compare queries with keys → normalize the scores into weights → use those weights to mix the values.**

This is the basic attention calculation. It does **not** include padding masks, causal masks, dropout, or multi-head splitting, so every query can attend to every supplied key.