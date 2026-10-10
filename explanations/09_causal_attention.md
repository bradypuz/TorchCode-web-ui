# Causal Self-Attention

This function performs attention while preventing a token from reading the future. Its arithmetic is the same as ordinary scaled dot-product attention until it replaces forbidden scores with negative infinity. The crucial question is which entries in the score matrix correspond to future positions.

## Orient the score matrix

`B` is batch size, `S` the shared sequence length, `D_k` query/key width, and `D_v` value width. Inputs `Q/K` are `(B,S,D_k)` and `V` is `(B,S,D_v)`. Output is `(B,S,D_v)`.

Scores have shape `(B,S,S)`. In each batch matrix, **row `i` is a query and column `j` is a key**. Position `i` may read key `j` when `j <= i`. The diagonal is permitted: a token can read itself.

## 1. Calculate the unmasked similarities

```python
d_k = K.size(-1)
scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
S = scores.size(-1)
```

Transposing keys changes `(B,S,D_k)` into `(B,D_k,S)`, so the product compares all query/key pairs. Dividing by `sqrt(D_k)` keeps score scale more consistent across feature widths. The final score axis is the key sequence length, which the reference uses to construct the square mask.

No future information has been removed yet. The next statements decide which comparisons softmax may use.

## 2. Mark the strict upper triangle

```python
mask = torch.triu(torch.ones(S, S, device=scores.device, dtype=torch.bool), diagonal=1)
```

`ones` initially creates a grid of `True` entries. `triu(..., diagonal=1)` keeps only entries strictly above the main diagonal and sets the others to `False`. Because column indices increase from left to right, entries above the diagonal have `j > i`: their keys are in the query's future.

For three positions, the mask is:

```text
                   key 0   key 1   key 2
query 0               .       X       X
query 1               .       .       X
query 2               .       .       .

. = allowed (False in mask)     X = forbidden (True in mask)
```

The mask is Boolean and created on the same device as the scores. `diagonal=0` would also block each token's own key, including every key for query zero, so the offset of one matters.

## 3. Remove future scores before softmax

```python
scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
weights = torch.softmax(scores, dim=-1)
return torch.bmm(weights, V)
```

`unsqueeze(0)` changes the mask from `(S,S)` to `(1,S,S)`. Its size-one batch dimension broadcasts the same causal rule across all `B` examples.

Softmax exponentiates negative infinity to zero. It then normalizes over the allowed keys, the last axis. If every unmasked score is zero, the first row has weights `[1,0,0]`, the second `[0.5,0.5,0]`, and the third `[1/3,1/3,1/3]`.

With scalar values `[2,4,8]`, the outputs are therefore `2`, `(2+4)/2 = 3`, and `(2+4+8)/3 = 14/3`. Masking after softmax without renormalizing would produce different, incorrectly shrunken answers.

```torchcode-visual
attention-mask
```

## Run a prefix-only averaging example

```python
# Example: Every query averages only its allowed prefix
Q = torch.zeros(1, 3, 2, requires_grad=True)
K = torch.zeros(1, 3, 2, requires_grad=True)
V = torch.tensor([[[2.], [4.], [8.]]], requires_grad=True)
mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
assert torch.equal(mask, torch.tensor([[False, True, True],
                                     [False, False, True],
                                     [False, False, False]]))
out = causal_attention(Q, K, V)
assert out.shape == (1, 3, 1)
assert torch.allclose(out, torch.tensor([[[2.], [3.], [14. / 3.]]]))
changed_values = V.detach().clone()
changed_values[:, 2] = 800.
changed_out = causal_attention(Q, K, changed_values)
assert torch.allclose(changed_out[:, :2], out[:, :2])
out.sum().backward()
assert torch.allclose(V.grad, torch.tensor([[[11. / 6.], [5. / 6.], [1. / 3.]]]))
assert Q.grad is not None and K.grad is not None
```

Changing the last value cannot affect the first two queries because both mask it. The first value's gradient is `1 + 1/2 + 1/3`: it contributes to all three outputs with those weights. The last value contributes only to the last query.

## Reference scope

The square mask assumes aligned, equal-length self-attention. It has no cached-position offset, padding mask, dropout, or projection layers, and it still computes dense quadratic scores. For finite nonempty sequences, every row keeps its own diagonal key, avoiding an entirely masked row. Autograd follows all unmasked attention operations.
