# Grouped Query Attention

Grouped query attention lets several query heads share one key/value head. Each query head can still form its own distribution, because its query features differ. The reference uses smaller K/V projections and then explicitly repeats their head representations to match the number of query heads.

## Name the dimensions

`B` is batch size, `S` sequence length, `D = d_model` model width, `H = num_heads` query heads, and `G = num_kv_heads` key/value heads. Head width is `d_k = D // H`. The intended setup requires positive counts, `D % H == 0`, and `H % G == 0`.

Input/output are `(B,S,D)`. Queries become `(B,H,S,d_k)`. Before repeating, keys and values are `(B,G,S,d_k)`. For `D=4`, `H=4`, and `G=2`, there is one feature per head, and each K/V head is shared by two query heads.

## 1. Give queries full width but K/V reduced width

```python
self.num_heads = num_heads
self.num_kv_heads = num_kv_heads
self.d_k = d_model // num_heads
self.W_q = nn.Linear(d_model, d_model)
self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k)
self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k)
self.W_o = nn.Linear(d_model, d_model)
```

In the small setup, `W_q` emits four features, while `W_k` and `W_v` emit only two. That is the structural saving: there are fewer distinct key/value feature vectors to compute and, in a cached implementation, to store. The default linear biases are included.

## 2. Reshape each projection into its own head count

```python
B, S, _ = x.shape
q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
k = self.W_k(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)
v = self.W_v(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)
```

`view` separates the final feature axis into head and within-head axes. `transpose(1,2)` changes token-major `(B,S,heads,d_k)` into head-major `(B,heads,S,d_k)`, so the following matrix products operate independently per head. This is self-attention: all three projections receive the same input stream.

## 3. Assign consecutive query heads to shared K/V heads

```python
repeats = self.num_heads // self.num_kv_heads
k = k.repeat_interleave(repeats, dim=1)
v = v.repeat_interleave(repeats, dim=1)
```

With four query heads and two K/V heads, `repeats` is two. `repeat_interleave` copies each original head twice before moving to the next head. It produces the mapping below; ordinary repeating in an alternating order would assign different groups.

| Query head | Shared K/V head | Example scalar values at two tokens |
|---:|---:|---|
| 0 | 0 | `[2,6]` |
| 1 | 0 | `[2,6]` |
| 2 | 1 | `[4,8]` |
| 3 | 1 | `[4,8]` |

After this step, Q/K/V all have `H` heads. The K/V copies are not new independently learned projections; gradients from their uses accumulate into their shared original parameters.

## 4. Attend, concatenate, and project

```python
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
weights = torch.softmax(scores, dim=-1)
attn = torch.matmul(weights, v)
out = attn.transpose(1, 2).contiguous().view(B, S, -1)
return self.W_o(out)
```

Scores have shape `(B,H,S,S)`. Softmax uses the last axis, key positions. Each query head can give different weights even when it shares K/V with another head. The output transpose restores `(B,S,H,d_k)`; `contiguous()` makes joining the last two axes with `view` safe. `W_o` mixes the concatenated features.

## Run a visible sharing example

Zero Q/K projections make every head average its two values. The V projection selects only input features zero and three, making the two shared groups easy to inspect. The expected joined output is `[4,4,6,6]` at both query positions.

```python
# Example: Four query heads share two value heads
gqa = GroupQueryAttention(d_model=4, num_heads=4, num_kv_heads=2)
with torch.no_grad():
    for projection in [gqa.W_q, gqa.W_k, gqa.W_v, gqa.W_o]:
        projection.weight.zero_()
        projection.bias.zero_()
    gqa.W_v.weight.copy_(torch.tensor([[1., 0., 0., 0.], [0., 0., 0., 1.]]))
    gqa.W_o.weight.copy_(torch.eye(4))
x = torch.tensor([[[2., 20., 200., 4.], [6., 60., 600., 8.]]], requires_grad=True)
small_values = gqa.W_v(x).view(1, 2, 2, 1).transpose(1, 2)
expanded = small_values.repeat_interleave(2, dim=1)
assert expanded.shape == (1, 4, 2, 1)
assert torch.equal(expanded[0, :, :, 0], torch.tensor([[2., 6.], [2., 6.], [4., 8.], [4., 8.]]))
out = gqa.forward(x)
assert torch.allclose(out, torch.tensor([[[4., 4., 6., 6.], [4., 4., 6., 6.]]]))
out.sum().backward()
assert torch.allclose(x.grad, torch.tensor([[[2., 0., 0., 2.], [2., 0., 0., 2.]]]))
```

Two heads and two queries each use half of a selected value, giving total input gradient `2`. Features one and two are unused by this deliberately fixed projection.

## Reference scope

This plain Python class has no automatic `nn.Module` container methods. It does not implement caching, a causal mask, or a memory-efficient grouped kernel: `repeat_interleave` explicitly copies heads and attention scores remain dense. When `G=H`, each group contains one head and the structure becomes ordinary multi-head self-attention. Invalid head counts are not checked explicitly.
