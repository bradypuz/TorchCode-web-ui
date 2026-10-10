# KV Cache Attention

A cache stores keys and values already projected for earlier tokens. New tokens still need their own queries, keys, and values, but old K/V projections can be reused. This function returns both the new-token output and the longer cache needed for the next call. The important subtlety is that new queries use local indices while cached keys use full-prefix indices.

## Name the changing sequence lengths

`B` is batch size, `D = d_model` model width, `H = num_heads` head count, and `d_k = D // H` head width. `S_new` is the number of tokens supplied in this call, `S_past` the cached token count, and `S_total = S_past + S_new`.

Input `x` is `(B,S_new,D)`. Cache is either `None` or `(K_past,V_past)`, each `(B,H,S_past,d_k)`. Output is `(B,S_new,D)`; returned cached keys/values are `(B,H,S_total,d_k)`. The function does not return outputs for past tokens again.

## 1. Create registered projections and project only new tokens

```python
super().__init__()
self.num_heads = num_heads
self.d_k = d_model // num_heads
self.W_q = nn.Linear(d_model, d_model)
self.W_k = nn.Linear(d_model, d_model)
self.W_v = nn.Linear(d_model, d_model)
self.W_o = nn.Linear(d_model, d_model)
```

The class inherits from `nn.Module`, so these child layers and their default biases are registered. Model width must be divisible by the positive head count.

```python
B, S_new, _ = x.shape
q = self.W_q(x).view(B, S_new, self.num_heads, self.d_k).transpose(1, 2)
k = self.W_k(x).view(B, S_new, self.num_heads, self.d_k).transpose(1, 2)
v = self.W_v(x).view(B, S_new, self.num_heads, self.d_k).transpose(1, 2)
```

`view` separates model features into head groups; `transpose` changes `(B,S_new,H,d_k)` into `(B,H,S_new,d_k)`. Only `x` is projected. Past keys and values will come from the cache; past queries are unnecessary because their outputs have already been produced.

## 2. Join cached and new keys/values on the token axis

```python
if cache is not None:
    k = torch.cat([cache[0], k], dim=2)
    v = torch.cat([cache[1], v], dim=2)
new_cache = (k, v)
S_total = k.shape[2]
```

Axis two is sequence length in head-major layout. Past tensors are listed first, preserving chronological order. `cat` creates longer tensors; it does not extend the old cache tensors in place. Returning `new_cache` lets the caller supply that longer state on the next call.

For two cached tokens and two new tokens, K/V length changes from two to four. Q length remains two: only the new positions ask questions.

## 3. Score new queries against the full available prefix

```python
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
```

Scores have shape `(B,H,S_new,S_total)`, usually rectangular. The query's local row zero may represent absolute token position two, three, or much later, depending on how much state is cached. A standard square causal mask starting at position zero would incorrectly block valid past keys.

## 4. Offset the causal mask when a chunk contains multiple new tokens

```python
if S_new > 1:
    S_past = S_total - S_new
    mask = torch.triu(
        torch.ones(S_new, S_total, device=x.device, dtype=torch.bool),
        diagonal=S_past + 1,
    )
    scores = scores.masked_fill(mask, float('-inf'))
```

New query row `i` has absolute position `S_past+i`. It may use key column `j` when `j <= S_past+i`. The upper-triangular offset `S_past+1` marks exactly the columns beyond that point.

```text
Two past tokens + two new tokens:
                         full key position
                         0   1   2   3
new query 0 (absolute 2)  .   .   .   X
new query 1 (absolute 3)  .   .   .   .

. = permitted     X = future inside the new chunk
```

With no cache, `S_past=0`, giving the ordinary `diagonal=1` mask. For a single new token, the mask is skipped: every assembled key is already at or before that query's position.

## 5. Gather values, merge heads, and return the state

```python
weights = torch.softmax(scores, dim=-1)
attn = torch.matmul(weights, v)
out = self.W_o(attn.transpose(1, 2).contiguous().view(B, S_new, -1))
return out, new_cache
```

Softmax normalizes across full-prefix keys. The value product returns `(B,H,S_new,d_k)`. Transpose restores token-major order; `contiguous()` supports flattening head features back into width `D` before `W_o`.

## Run full, chunked, and single-token calls

Zero Q/K projections make permitted keys equally weighted. Identity V/output projections expose prefix averages of tokens `[2,20]`, `[4,40]`, `[8,80]`, and `[16,160]`. Those averages are `[2,20]`, `[3,30]`, `[14/3,140/3]`, and `[7.5,75]` regardless of whether the tokens arrive together or in chunks.

```python
# Example: A cache offset preserves the same prefix averages
model = KVCacheAttention(d_model=2, num_heads=1)
with torch.no_grad():
    for projection in [model.W_q, model.W_k, model.W_v, model.W_o]:
        projection.weight.zero_()
        projection.bias.zero_()
    model.W_v.weight.copy_(torch.eye(2))
    model.W_o.weight.copy_(torch.eye(2))
x = torch.tensor([[[2., 20.], [4., 40.], [8., 80.], [16., 160.]]], requires_grad=True)
full, _ = model(x)
expected = torch.tensor([[[2., 20.], [3., 30.], [14. / 3., 140. / 3.], [7.5, 75.]]])
assert torch.allclose(full, expected)
prefill, cache = model(x[:, :2])
assert cache[0].shape == (1, 1, 2, 2)
offset_mask = torch.triu(torch.ones(2, 4, dtype=torch.bool), diagonal=3)
assert torch.equal(offset_mask, torch.tensor([[False, False, False, True],
                                            [False, False, False, False]]))
chunk, longer_cache = model(x[:, 2:], cache=cache)
assert longer_cache[0].shape == (1, 1, 4, 2)
assert cache[0].shape[2] == 2
joined = torch.cat([prefill, chunk], dim=1)
assert torch.allclose(joined, expected)
third, one_more = model(x[:, 2:3], cache=cache)
fourth, final_cache = model(x[:, 3:4], cache=one_more)
assert torch.allclose(torch.cat([prefill, third, fourth], dim=1), expected)
assert final_cache[1].shape[2] == 4
joined.sum().backward()
expected_grad = torch.tensor([[[25. / 12., 25. / 12.], [13. / 12., 13. / 12.],
                               [7. / 12., 7. / 12.], [0.25, 0.25]]])
assert torch.allclose(x.grad, expected_grad)
```

The gradient check also shows that cached values stay connected to earlier inputs. The reference does not detach state automatically.

## Reference scope

Caches must belong to the same sequence and unchanged model, with compatible batch/head shapes, device, and dtype. Repeated concatenation copies the growing prefix, so this is not a preallocated inference cache. No padding mask, positional offsets for embeddings, or validation layer is provided; the offset here controls attention visibility only.
