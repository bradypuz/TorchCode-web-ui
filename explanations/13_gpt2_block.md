This implements a **GPT-2-style Transformer block**: it takes a sequence of token representations, lets each token gather information from earlier tokens, then processes each token through a small neural network.

It is **one block**, not a complete GPT-2 model. A full model would also include token and position embeddings, a stack of these blocks, and a final layer that predicts tokens.

## 1. Input and output

The input `x` has shape:

```text
(B, S, d_model)
```

where:

- **B** = batch size, or number of sequences processed together.
- **S** = sequence length.
- **d_model** = number of features representing each token.

For example, with `B=2`, `S=10`, and `d_model=768`, the input contains two sequences of ten tokens, with each token represented by a 768-dimensional vector.

The output has **the same shape**. The block updates the token representations without changing their number or size.

---

## 2. Initialization: defining the layers

### Attention heads

```python
self.num_heads = num_heads
self.d_k = d_model // num_heads
```

Multi-head attention divides the representation into several attention heads. Each head can learn different relationships between tokens.

For example:

```text
d_model = 768
num_heads = 12
d_k = 64
```

Each head works with 64-dimensional queries, keys, and values.

**Important:** `d_model` must be divisible by `num_heads`; otherwise, the reshaping in `_attn` will fail.

### Layer normalization

```python
self.ln1 = nn.LayerNorm(d_model)
self.ln2 = nn.LayerNorm(d_model)
```

These normalize each token’s feature vector across its `d_model` dimensions, with learnable scale and bias parameters.

There are two separate normalization layers:

- `ln1`: before attention.
- `ln2`: before the feed-forward network.

This helps keep activations well behaved during training.

### Attention projections

```python
self.W_q = nn.Linear(d_model, d_model)
self.W_k = nn.Linear(d_model, d_model)
self.W_v = nn.Linear(d_model, d_model)
self.W_o = nn.Linear(d_model, d_model)
```

These are learned transformations:

- **Query (`q`)**: what information this token is looking for.
- **Key (`k`)**: what information a token can be matched on.
- **Value (`v`)**: the information a token contributes when attended to.
- **Output projection (`W_o`)**: combines the attention heads into a new token representation.

### Feed-forward network

```python
self.mlp = nn.Sequential(
    nn.Linear(d_model, 4 * d_model),
    nn.GELU(),
    nn.Linear(4 * d_model, d_model),
)
```

This transforms each token independently:

```text
d_model → 4 × d_model → d_model
```

The first layer expands the representation, `GELU` introduces nonlinearity, and the final layer projects it back to its original size.

The **same MLP weights are used at every sequence position**.

---

## 3. `_attn`: causal multi-head self-attention

### Step A: Create queries, keys, and values

```python
B, S, _ = x.shape
q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
```

The same operation is performed for `k` and `v`.

For each tensor, the shape changes as follows:

```text
After linear projection: (B, S, d_model)
After view:              (B, S, num_heads, d_k)
After transpose:         (B, num_heads, S, d_k)
```

The `view` splits the feature dimension into heads. The `transpose` arranges the tensor so attention can be computed separately for every head.

### Step B: Compute attention scores

```python
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
```

This implements:

\[
\text{scores} = \frac{QK^\top}{\sqrt{d_k}}
\]

Each query is compared with every key using a dot product.

The result has shape:

```text
(B, num_heads, S, S)
```

Here, `scores[b, h, i, j]` measures how strongly token **i** should attend to token **j** in head **h**.

Dividing by `sqrt(d_k)` helps prevent the scores from becoming too large and making the softmax overly concentrated.

### Step C: Hide future tokens

```python
mask = torch.triu(
    torch.ones(S, S, device=x.device, dtype=torch.bool),
    diagonal=1
)
scores = scores.masked_fill(mask, float('-inf'))
```

This creates a **causal mask**. A token can attend to itself and earlier tokens, but not future tokens.

For four tokens, the allowed attention pattern is:

```text
                 Key token
                 0  1  2  3
Query token 0    ✓  ✗  ✗  ✗
Query token 1    ✓  ✓  ✗  ✗
Query token 2    ✓  ✓  ✓  ✗
Query token 3    ✓  ✓  ✓  ✓
```

The mask has shape `(S, S)` and broadcasts across batches and heads.

Masked scores become `-inf`, which gives them zero probability after softmax. This prevents the model from “looking ahead” when predicting text.

### Step D: Turn scores into weights and gather information

```python
weights = torch.softmax(scores, dim=-1)
attn = torch.matmul(weights, v)
```

Softmax converts each query’s scores into attention weights over the key positions. Those weights sum to 1 along the last dimension.

The weighted sum of values is:

\[
\text{Attention}(Q,K,V)
=
\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt{d_k}} + \text{causal mask}
\right)V
\]

The result has shape:

```text
(B, num_heads, S, d_k)
```

Each token now contains information gathered from its permitted context.

### Step E: Recombine the heads

```python
return self.W_o(
    attn.transpose(1, 2).contiguous().view(B, S, -1)
)
```

The shape changes are:

```text
(B, num_heads, S, d_k)
          ↓ transpose
(B, S, num_heads, d_k)
          ↓ combine heads
(B, S, d_model)
```

`contiguous()` makes the memory layout suitable for the following `view`. The `-1` tells PyTorch to infer the final dimension, which is `d_model`.

Finally, `W_o` learns how to mix the combined head outputs.

---

## 4. `forward`: normalization and residual connections

```python
def forward(self, x):
    x = x + self._attn(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x
```

This contains two stages.

### Stage 1: Attention

```python
x = x + self._attn(self.ln1(x))
```

1. Normalize the input.
2. Run causal self-attention.
3. Add the attention result to the original input.

The addition is a **residual connection**: the attention layer adds an update instead of having to reconstruct the entire representation.

### Stage 2: Feed-forward processing

```python
x = x + self.mlp(self.ln2(x))
```

1. Normalize the updated representation.
2. Apply the MLP to each token.
3. Add the MLP result through another residual connection.

In equations:

\[
x_1 = x + \operatorname{Attention}(\operatorname{LN}_1(x))
\]

\[
x_{\text{out}} = x_1 + \operatorname{MLP}(\operatorname{LN}_2(x_1))
\]

Because normalization happens **before** each sublayer, this is called a **pre-LayerNorm**, or **pre-LN**, block.

## The key intuition

- **Attention mixes information across tokens**, while respecting the causal mask.
- **The MLP processes the features within each token**.
- **Residual connections preserve the existing representation and support gradient flow**.
- **Layer normalization helps stabilize training**.

This captures GPT-2’s core block structure, though it omits details such as dropout and does not include embeddings or a token-prediction head.