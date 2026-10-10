# GPT-2 Transformer Block

This class combines two transformations. Causal attention lets a token read earlier tokens. An MLP transforms that token's features. Each transformation sits on a normalized branch whose output is added to the current representation. Reading those additions in order explains what “pre-norm residual block” means in the actual code.

## Keep track of the dimensions

`B` is batch size, `S` sequence length, `D = d_model` model width, and `H = num_heads` attention heads. Per-head width is `d_k = D // H`, requiring divisible dimensions. Input and output are `(B,S,D)`.

| Stage | Shape |
|---|---|
| Input and LayerNorm outputs | `(B,S,D)` |
| Q/K/V split into heads | `(B,H,S,d_k)` |
| Attention scores | `(B,H,S,S)` |
| Concatenated attention result | `(B,S,D)` |
| Expanded MLP hidden features | `(B,S,4D)` |
| MLP result and every residual sum | `(B,S,D)` |

Neither residual addition can work if its branch returns a different feature width.

## 1. Register the components

```python
super().__init__()
self.num_heads = num_heads
self.d_k = d_model // num_heads
self.ln1 = nn.LayerNorm(d_model)
self.ln2 = nn.LayerNorm(d_model)
```

The class inherits from `nn.Module`. Calling `super().__init__()` enables registration of its child layers and parameters, so `.parameters()`, device transfers, and saved state include them. `ln1` and `ln2` are distinct LayerNorms, each normalizing features within one token. Their learned scales and shifts can differ.

```python
self.W_q = nn.Linear(d_model, d_model)
self.W_k = nn.Linear(d_model, d_model)
self.W_v = nn.Linear(d_model, d_model)
self.W_o = nn.Linear(d_model, d_model)
self.mlp = nn.Sequential(
    nn.Linear(d_model, 4 * d_model),
    nn.GELU(),
    nn.Linear(4 * d_model, d_model),
)
```

The four attention projections preserve width. The MLP first expands width to `4D`, applies a smooth elementwise GELU nonlinearity, then returns to `D`. `nn.Sequential` applies those steps in their listed order. The MLP acts independently at each token; it does not mix sequence positions.

## 2. Compute attention on a supplied normalized tensor

Inside `_attn`, the projections are split into heads:

```python
B, S, _ = x.shape
q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
k = self.W_k(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
v = self.W_v(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
```

`view` groups adjacent projected features into heads. The transpose changes `(B,S,H,d_k)` into `(B,H,S,d_k)`, allowing one matrix product per batch/head.

```python
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(mask, float('-inf'))
weights = torch.softmax(scores, dim=-1)
attn = torch.matmul(weights, v)
return self.W_o(attn.transpose(1, 2).contiguous().view(B, S, -1))
```

Rows are queries and columns are keys. The strict upper triangle marks future keys. Its `(S,S)` mask broadcasts across both batch and head axes. Negative infinity makes forbidden softmax weights zero; the last-axis softmax normalizes across allowed keys. The value product returns `(B,H,S,d_k)`. Transposing back, making storage contiguous, and joining head features restores `(B,S,D)` before `W_o` mixes them.

## 3. Read the two residual updates in order

```python
x = x + self._attn(self.ln1(x))
x = x + self.mlp(self.ln2(x))
return x
```

In the first statement, only the attention branch receives `ln1(x)`. The residual path still carries the original `x`. After that addition, the variable `x` has changed. Both the second residual path and `ln2` now use this **attention-updated** tensor, not the original input.

```text
original x ───────────────────┐
    └→ ln1 → causal attention ├→ add → updated x ───────────┐
                              ┘       └→ ln2 → MLP ────────├→ add → output
                                                          ┘
```

“Pre-norm” describes this placement of LayerNorm before each branch. Residual paths provide direct routes for information and gradients around the transformations.

## Run an addition you can inspect numerically

For easy arithmetic, set Q/K to zero, V/output to identity, and make the MLP return a fixed output bias. Input tokens `[1,3,1,3]` and `[2,6,2,6]` both normalize approximately to `[-1,1,-1,1]`. Causal uniform attention gives the first token its own normalized value and the second the average of both. Attention residuals are approximately `[0,4,0,4]` and `[1,7,1,7]`. Adding the MLP bias then gives the assertions below. LayerNorm epsilon accounts for tiny deviations from these rounded numbers.

```python
# Example: See both residual additions in one small block
block = GPT2Block(d_model=4, num_heads=2)
with torch.no_grad():
    for parameter in block.parameters():
        parameter.zero_()
    block.ln1.weight.fill_(1.)
    block.ln2.weight.fill_(1.)
    block.W_v.weight.copy_(torch.eye(4))
    block.W_o.weight.copy_(torch.eye(4))
    block.mlp[-1].bias.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))
x = torch.tensor([[[1., 3., 1., 3.], [2., 6., 2., 6.]]], requires_grad=True)
normalized = block.ln1(x)
assert torch.allclose(normalized, torch.tensor([[[-1., 1., -1., 1.], [-1., 1., -1., 1.]]]), atol=1e-4)
out = block(x)
expected = torch.tensor([[[0.1, 4.2, 0.3, 4.4], [1.1, 7.2, 1.3, 7.4]]])
assert out.shape == (1, 2, 4)
assert torch.allclose(out, expected, atol=1e-4)
changed = x.detach().clone()
changed[:, 1] = torch.tensor([100., -100., 20., -20.])
assert torch.allclose(block(changed)[:, 0], out[:, 0])
out.sum().backward()
assert x.grad is not None
assert all(parameter.grad is not None for parameter in block.parameters())
```

Changing the future token leaves the first result unchanged because LayerNorm/MLP are tokenwise and the attention mask blocks future access.

## Reference scope

This is one simplified decoder block, not a complete language model. It has no embeddings, vocabulary head, dropout, positional encoding, or KV cache. Attention remains dense, and head divisibility is assumed rather than validated.
