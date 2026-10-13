This code implements **multi-head scaled dot-product attention**, the mechanism used in Transformers.

Its job is to let each position in a sequence gather information from other positions, with different attention heads learning different ways to decide what is relevant.

## 1. What are `Q`, `K`, and `V`?

Attention uses three inputs:

- **Queries (`Q`)**: what each position is looking for.
- **Keys (`K`)**: what each available position can be matched against.
- **Values (`V`)**: the information retrieved from those positions.

For each query, attention:
1. Compares it with every key.
2. Converts those comparisons into weights.
3. Computes a weighted average of the values.

For example, if the attention weights for a query are:

```text
[0.1, 0.7, 0.2]
```

then its output is:

```text
0.1 × value_1 + 0.7 × value_2 + 0.2 × value_3
```

In **self-attention**, all three inputs usually start as the same sequence:

```python
attention.forward(x, x, x)
```

The learned projections then turn that sequence into different queries, keys, and values.

In **cross-attention**, `Q` can come from one sequence while `K` and `V` come from another.

---

## 2. Initialization

```python
def __init__(self, d_model: int, num_heads: int):
    self.num_heads = num_heads
    self.d_k = d_model // num_heads
```

- `d_model` is the size of each input token’s feature vector.
- `num_heads` is the number of attention heads.
- `d_k` is the feature size used by each head.

For example:

```text
d_model = 512
num_heads = 8
d_k = 64
```

Each head works with 64-dimensional queries, keys, and values.

**`d_model` must be divisible by `num_heads`** for the reshaping below to work.

### Learned projections

```python
self.W_q = nn.Linear(d_model, d_model)
self.W_k = nn.Linear(d_model, d_model)
self.W_v = nn.Linear(d_model, d_model)
self.W_o = nn.Linear(d_model, d_model)
```

These are four independently learned linear layers:

| Layer | Purpose |
|---|---|
| `W_q` | Convert input features into query features |
| `W_k` | Convert input features into key features |
| `W_v` | Convert input features into value features |
| `W_o` | Mix the results from all heads into the final output |

`nn.Linear` operates on the **last dimension**, so these layers preserve the batch and sequence dimensions.

Although the features are split into heads afterward, each projection initially operates on the full input feature vector. The heads are therefore not simply restricted to fixed chunks of the original input.

---

## 3. Read the input dimensions

```python
B, S_q, _ = Q.shape
S_k = K.shape[1]
```

The expected input shapes are:

```text
Q: [B, S_q, d_model]
K: [B, S_k, d_model]
V: [B, S_k, d_model]
```

Here:

- `B`: batch size.
- `S_q`: number of query positions.
- `S_k`: number of key/value positions.

`S_q` and `S_k` do not have to be equal. For example, a decoder could have 5 query positions attending to 10 encoder positions.

`K` and `V` must have the same sequence length because each key corresponds to one value.

---

## 4. Project the inputs and split them into heads

```python
q = self.W_q(Q).view(
    B, S_q, self.num_heads, self.d_k
).transpose(1, 2)
```

There are three operations here.

### A. Project into query features

```python
self.W_q(Q)
```

Shape:

```text
[B, S_q, d_model]
```

### B. Split the feature dimension into heads

```python
.view(B, S_q, self.num_heads, self.d_k)
```

Shape:

```text
[B, S_q, num_heads, d_k]
```

For instance, a feature dimension of 512 becomes 8 heads with 64 features each.

### C. Move the head dimension before the sequence dimension

```python
.transpose(1, 2)
```

Shape:

```text
[B, num_heads, S_q, d_k]
```

This arrangement lets PyTorch perform attention separately for every batch item and every head.

The same process is applied to keys and values:

```python
k = self.W_k(K).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)
v = self.W_v(V).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)
```

Resulting shapes:

```text
q: [B, num_heads, S_q, d_k]
k: [B, num_heads, S_k, d_k]
v: [B, num_heads, S_k, d_k]
```

---

## 5. Compute attention scores

```python
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
```

First, this swaps the last two dimensions of `k`:

```python
k.transpose(-2, -1)
```

```text
Before: [B, num_heads, S_k, d_k]
After:  [B, num_heads, d_k, S_k]
```

The matrix multiplication then has the form:

```text
[S_q, d_k] @ [d_k, S_k] → [S_q, S_k]
```

It happens independently for every batch item and head, producing:

```text
scores: [B, num_heads, S_q, S_k]
```

Each score is a dot product between one query and one key:

\[
\text{score}_{i,j}=\frac{q_i\cdot k_j}{\sqrt{d_k}}
\]

A larger score means the model considers that key more relevant to that query.

### Why divide by `sqrt(d_k)`?

Dot products tend to have larger magnitudes as the feature dimension grows. Very large scores can make softmax excessively peaked, which can make optimization harder.

Dividing by \(\sqrt{d_k}\) helps keep the scale of the scores under control.

---

## 6. Convert scores into attention weights

```python
weights = torch.softmax(scores, dim=-1)
```

The last dimension is `S_k`, so softmax runs across all keys **for each query, in each head**.

For example:

```text
Scores:  [1.0, 3.0, 2.0]
Weights: [0.09, 0.67, 0.24]  # approximately
```

The weights are nonnegative and sum to 1 across the key positions.

The shape remains:

```text
weights: [B, num_heads, S_q, S_k]
```

---

## 7. Take a weighted sum of the values

```python
attn = torch.matmul(weights, v)
```

The matrix multiplication is:

```text
[S_q, S_k] @ [S_k, d_k] → [S_q, d_k]
```

Therefore:

```text
attn: [B, num_heads, S_q, d_k]
```

For a particular query position \(i\), the result is:

\[
\text{attn}_i=\sum_{j=1}^{S_k}\text{weights}_{i,j}\,v_j
\]

This is the information that query gathers from the value sequence.

Each head produces its own result because it has its own projected queries, keys, and values.

---

## 8. Combine the heads

```python
out = attn.transpose(1, 2).reshape(B, S_q, -1)
```

First:

```text
[B, num_heads, S_q, d_k]
                  ↓ transpose(1, 2)
[B, S_q, num_heads, d_k]
```

Then `reshape` merges the final two dimensions:

```text
[B, S_q, num_heads × d_k]
```

Since:

```text
num_heads × d_k = d_model
```

the result is:

```text
out: [B, S_q, d_model]
```

This is effectively **concatenating the heads’ outputs** for each query position.

Using `reshape` here is useful because the preceding `transpose` may produce a non-contiguous tensor; unlike `view`, `reshape` can make a copy when necessary.

---

## 9. Apply the output projection

```python
return self.W_o(out)
```

This learned linear layer mixes information across the concatenated heads.

The final output shape is:

```text
[B, S_q, d_model]
```

Notice that the output sequence length is the **query length**, because attention produces one output vector per query.

## Shape summary

Let `H = num_heads` and `D = d_model`.

| Tensor | Shape |
|---|---|
| Input `Q` | `[B, S_q, D]` |
| Input `K`, `V` | `[B, S_k, D]` |
| Projected and split `q` | `[B, H, S_q, d_k]` |
| Projected and split `k`, `v` | `[B, H, S_k, d_k]` |
| `scores` | `[B, H, S_q, S_k]` |
| `weights` | `[B, H, S_q, S_k]` |
| `attn` | `[B, H, S_q, d_k]` |
| Combined heads | `[B, S_q, D]` |
| Final output | `[B, S_q, D]` |

## Important issues in this implementation

### 1. It should inherit from `nn.Module`

For normal PyTorch usage, the class should begin like this:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        if num_heads <= 0 or d_model <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be positive and divisible by num_heads")

        # Rest of initialization...
```

Inheriting from `nn.Module` and calling `super().__init__()` registers the layers and enables standard functionality such as:

```python
model.parameters()
model.to(device)
model.state_dict()
model(Q, K, V)
```

The original code can still compute outputs and gradients through its individual linear layers, but it lacks this standard module interface and automatic registration.

### 2. There is no attention mask

As written, every query can attend to every key.

That means it does **not**:
- Ignore padding automatically.
- Prevent attention to future tokens in causal self-attention.

A mask would normally modify `scores` **before softmax**.

### 3. This is attention, not a full Transformer block

Residual connections, layer normalization, a feed-forward network, and dropout are not included.

**In short: the code learns query–key similarities, uses them to gather value information in several heads, concatenates the results, and projects them back to `d_model` dimensions.**