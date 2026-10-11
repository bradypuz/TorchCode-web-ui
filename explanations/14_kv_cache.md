This implements **multi-head causal self-attention with a key/value (KV) cache**—the mechanism commonly used to make autoregressive generation more efficient.

The central idea is:

> Compute queries, keys, and values for the **new tokens**, reuse the keys and values from **previous tokens**, and prevent each new token from attending to future tokens.

## 1. What goes in and comes out?

```python
out, new_cache = attention(x, cache)
```

- **`x`**: embeddings or hidden states for the new tokens, shaped **`[B, S_new, d_model]`**.
- **`cache`**: optionally, a tuple `(past_keys, past_values)` from earlier calls.
- **`out`**: attention outputs for only the new tokens, shaped **`[B, S_new, d_model]`**.
- **`new_cache`**: keys and values for all tokens seen so far, including the new ones.

Here:

| Symbol | Meaning |
|---|---|
| `B` | Batch size |
| `S_new` | Number of tokens supplied in the current call |
| `S_past` | Number of previously cached tokens |
| `S_total` | `S_past + S_new` |
| `num_heads` | Number of attention heads |
| `d_k` | Dimension of each head: `d_model // num_heads` |

For example, if `d_model = 512` and `num_heads = 8`, each head operates on 64-dimensional vectors.

---

## 2. Initialization: learn the projections

```python
self.num_heads = num_heads
self.d_k = d_model // num_heads
self.W_q = nn.Linear(d_model, d_model)
self.W_k = nn.Linear(d_model, d_model)
self.W_v = nn.Linear(d_model, d_model)
self.W_o = nn.Linear(d_model, d_model)
```

Each token’s input vector is projected into three different representations:

- **Query (`q`)**: what information this token is looking for.
- **Key (`k`)**: what information this token can be matched against.
- **Value (`v`)**: the information contributed if this token receives attention.

`W_o` is the output projection applied after combining the attention heads.

These are trainable linear layers. The implementation assumes that **`d_model` is divisible by `num_heads`**; otherwise, the later reshaping fails.

## 3. Project the new tokens and split them into heads

```python
B, S_new, _ = x.shape

q = self.W_q(x).view(B, S_new, self.num_heads, self.d_k).transpose(1, 2)
k = self.W_k(x).view(B, S_new, self.num_heads, self.d_k).transpose(1, 2)
v = self.W_v(x).view(B, S_new, self.num_heads, self.d_k).transpose(1, 2)
```

Each projection follows this shape transformation:

```text
[B, S_new, d_model]
          ↓ split the feature dimension into heads
[B, S_new, num_heads, d_k]
          ↓ transpose sequence and head dimensions
[B, num_heads, S_new, d_k]
```

This places the tensors in a layout suitable for computing attention independently in every head.

All three tensors initially describe **only the new tokens**.

## 4. Add previously cached keys and values

```python
if cache is not None:
    k = torch.cat([cache[0], k], dim=2)
    v = torch.cat([cache[1], v], dim=2)
```

Dimension `2` is the sequence dimension, so this appends the new keys and values after the old ones.

After concatenation:

```text
q: [B, num_heads, S_new,   d_k]
k: [B, num_heads, S_total, d_k]
v: [B, num_heads, S_total, d_k]
```

Notice that **queries are not cached**.

Each call computes outputs only for the new tokens, so only their queries are needed. Those new queries can attend to both past and current keys and values.

```python
new_cache = (k, v)
S_total = k.shape[2]
```

The combined keys and values become the cache for the next call.

### Why this helps

When generating the next token, earlier tokens’ keys and values have already been computed. Caching avoids recomputing those projections—and, in a full transformer, avoids rerunning the previous tokens through all the layers.

The new token must still attend over the growing history, so caching does **not** make attention cost independent of sequence length.

## 5. Compute attention scores

```python
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
```

This implements scaled dot-product attention scores:

\[
\text{scores} = \frac{QK^\top}{\sqrt{d_k}}
\]

The shapes are:

```text
q:                 [B, num_heads, S_new, d_k]
k.transpose(...):  [B, num_heads, d_k, S_total]
scores:            [B, num_heads, S_new, S_total]
```

For every new query token, there is one score for every available key token.

The division by \(\sqrt{d_k}\) helps keep the scores from becoming excessively large as the head dimension increases, which would make softmax overly concentrated.

## 6. Apply the causal mask

```python
if S_new > 1:
    S_past = S_total - S_new
    mask = torch.triu(
        torch.ones(S_new, S_total, device=x.device, dtype=torch.bool),
        diagonal=S_past + 1,
    )
    scores = scores.masked_fill(mask, float('-inf'))
```

**Causal attention means a token may attend to itself and earlier tokens, but not later tokens.**

The complication is that the queries cover only the new tokens, while the keys cover the entire history. The mask therefore needs an offset.

For new query index `i`, its position in the full sequence is:

```text
S_past + i
```

It may attend to key position `j` only when:

```text
j <= S_past + i
```

The shifted upper-triangular mask blocks exactly the positions where:

```text
j > S_past + i
```

### Example: three cached tokens and two new tokens

Suppose the sequence is:

```text
Cached: A B C
New:    D E
```

The allowed attention pattern is:

```text
                  Keys
              A  B  C  D  E
Query D       ✓  ✓  ✓  ✓  ✗
Query E       ✓  ✓  ✓  ✓  ✓
```

Here, `S_past = 3`, so `diagonal = 4`. That masks the score from `D` to `E`.

Masked scores become `-inf`, which turns into zero probability after softmax.

### Why skip the mask when `S_new == 1`?

With one new token, all keys belong to either:

- earlier cached tokens, or
- the current token itself.

There are no future tokens to block.

With no cache, `S_past = 0`, so the mask becomes the usual causal upper triangle.

## 7. Convert scores into weights and combine values

```python
weights = torch.softmax(scores, dim=-1)
attn = torch.matmul(weights, v)
```

Softmax runs across the key positions, converting each query’s scores into attention weights.

Then each query receives a weighted sum of the values:

\[
\text{Attention}(Q,K,V)
=
\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt{d_k}} + M
\right)V
\]

where \(M\) is zero for allowed positions and \(-\infty\) for blocked positions.

Shapes:

```text
weights: [B, num_heads, S_new, S_total]
v:       [B, num_heads, S_total, d_k]
attn:    [B, num_heads, S_new, d_k]
```

## 8. Combine the heads and produce the output

```python
out = self.W_o(
    attn.transpose(1, 2).contiguous().view(B, S_new, -1)
)
```

The shape changes are:

```text
[B, num_heads, S_new, d_k]
          ↓ transpose
[B, S_new, num_heads, d_k]
          ↓ concatenate head outputs
[B, S_new, d_model]
          ↓ output projection W_o
[B, S_new, d_model]
```

`.contiguous()` ensures the transposed tensor has a memory layout compatible with `.view()`.

Finally:

```python
return out, new_cache
```

## Typical usage

```python
attention = KVCacheAttention(d_model=128, num_heads=8)
attention.eval()

with torch.inference_mode():
    # Prefill: process an entire prompt with causal attention.
    prompt = torch.randn(1, 5, 128)
    prompt_out, cache = attention(prompt)

    # Decode: process one additional token using the cached history.
    next_token = torch.randn(1, 1, 128)
    next_out, cache = attention(next_token, cache)
```

After prefill, each cached tensor has shape:

```text
[1, 8, 5, 16]
```

After processing the next token:

```text
[1, 8, 6, 16]
```

**In short:** the new tokens supply the queries; all tokens seen so far supply the keys and values; the causal mask blocks future tokens; and the updated KV cache is carried into the next call.

This is an attention component, not a complete transformer block: it does not include positional encoding, normalization, residual connections, or a feed-forward network.