# Embedding lookup

## What the layer does

Replace each integer ID with a learned vector. For words, an ID identifies a vocabulary entry; the corresponding row of the table contains that word's features. The operation is a lookup, not multiplication by the numeric ID. ID `2` means “take row 2,” not “double another row.”

`num_embeddings` is the number of rows, and `embedding_dim` is the number of numbers in each row. The weight table has shape `(num_embeddings, embedding_dim)`. If `indices` has shape `(B, S)`, the output has shape `(B, S, embedding_dim)`, where `B` is the batch size and `S` the number of IDs per sequence. A 1-D input `(S,)` produces `(S, embedding_dim)`. In general, the lookup appends one vector dimension to the indices' shape.

## Create a trainable table

```python
super().__init__()
self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
```

`torch.randn(...)` fills the table with random values from a normal distribution centered at zero. These initial vectors have no vocabulary meaning yet. Learning changes them.

Wrapping the tensor in `nn.Parameter` makes it a registered trainable parameter. It will appear in the module's parameter list, and it requests gradients by default. Without that wrapper, an ordinary tensor assigned as `self.weight` would not automatically be included in an optimizer constructed from `model.parameters()`.

## Let tensor indexing do the lookup

```python
return self.weight[indices]
```

For `indices = [2, 0, 2]`, this returns rows 2, 0, and 2 in exactly that order. Both appearances of row 2 have the same values. The operation does not sort IDs or remove duplicates.

For a matrix of IDs, every coordinate receives its own row vector. If `indices[1, 0]` is `2`, then `output[1, 0, :]` is `weight[2, :]`. The extra final dimension represents features; the existing dimensions still represent the input's arrangement of tokens.

## Trace a lookup and repeated-row gradients

Use a three-row table with two features per row:

| ID | Weight row | Positions requesting it in `[[2, 0], [2, 1]]` |
|---|---|---|
| `0` | `[10, 11]` | `(0, 1)` |
| `1` | `[20, 21]` | `(1, 1)` |
| `2` | `[30, 31]` | `(0, 0)` and `(1, 0)` |

The output arrangement is:

```text
indices: (2, 2)                 output: (2, 2, 2)
[[2, 0],                        [[[30, 31], [10, 11]],
 [2, 1]]                         [[30, 31], [20, 21]]]
```

The repeated lookup matters during learning. Suppose the loss is the sum of every output number. Each appearance contributes a gradient `[1, 1]` to the table row it read. Row 2 is read twice, so these contributions add to `[2, 2]`. Rows 0 and 1 receive `[1, 1]` each. A row not selected anywhere would receive zero gradient for this loss.

```torchcode-visual
embedding
```

Change the selected IDs in the visual to see how output vectors repeat and gradient counts accumulate at their source rows.

```python
# Example: Look up a grid of IDs and accumulate repeated-row gradients
layer = MyEmbedding(num_embeddings=3, embedding_dim=2)
with torch.no_grad():
    layer.weight.copy_(torch.tensor([[10., 11.], [20., 21.], [30., 31.]]))
indices = torch.tensor([[2, 0], [2, 1]], dtype=torch.long)
out = layer(indices)
expected = torch.tensor([[[30., 31.], [10., 11.]], [[30., 31.], [20., 21.]]])
assert out.shape == (2, 2, 2)
assert torch.equal(out, expected)
out.sum().backward()
assert torch.equal(layer.weight.grad, torch.tensor([[1., 1.], [1., 1.], [2., 2.]]))
assert torch.equal(layer(torch.tensor([1, 1])), torch.tensor([[20., 21.], [20., 21.]]))
```

The second lookup shows the same rule for a 1-D index tensor. The table still has just one row for ID 1; repeating an ID does not create additional trainable rows. Gradients go to the floating-point table, while integer IDs themselves are not differentiable.

## Common mistakes and reference scope

Use integer indices, normally `torch.long`, and keep vocabulary IDs within `0` through `num_embeddings - 1`. Direct PyTorch indexing accepts negative indices by counting backward from the table's end; that is a behavior of this reference, not usually a desirable vocabulary convention. There is no special padding row, sparse-gradient option, or automatic rescaling of frequently repeated IDs. Repetition simply adds gradient contributions.
