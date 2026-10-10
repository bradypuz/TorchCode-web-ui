# SwiGLU MLP

## What the code is trying to do

For each input vector, create some intermediate features, let a second set of features control their strength, and combine the result into an output vector. In a transformer these vectors usually represent tokens. This layer processes each token independently: changing token 2 does not directly change token 1's output.

`d_model` is the input and output width. `d_ff` is the intermediate width, often larger than `d_model`. An input of shape `(B, S, d_model)` produces an output of shape `(B, S, d_model)`, where `B` is the batch size and `S` the number of tokens. A `(B, d_model)` input also works. Both branches inside the layer have shape `(..., d_ff)`; `...` means all the input's leading dimensions.

## Build the three learned projections

```python
super().__init__()
self.gate_proj = nn.Linear(d_model, d_ff)
self.up_proj = nn.Linear(d_model, d_ff)
self.down_proj = nn.Linear(d_ff, d_model)
```

Calling `super().__init__()` initializes PyTorch's module machinery so assigned layers become registered submodules. Their parameters then appear in `model.parameters()` and are available to an optimizer.

Each `nn.Linear(a, b)` accepts a vector of length `a` and returns one of length `b`. Its weight has shape `(b, a)`, and it computes `x @ weight.T + bias`. Biases are enabled in this reference. `gate_proj` and `up_proj` have identical dimensions but separate parameters, so they can learn different transformations. `down_proj` combines the intermediate features and restores the original width.

## Follow one vector through `forward`

The reference compresses the whole forward pass into one statement:

```python
return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

Read it from the inside out. `gate_proj(x)` first creates a gate value for each intermediate feature. `F.silu(z)` computes `z * sigmoid(z)`, where `sigmoid(z) = 1 / (1 + exp(-z))`. For example, SiLU changes `1` into about `0.7311`, changes `0` into `0`, and changes `-1` into about `-0.2689`.

`up_proj(x)` creates the other branch from the **original** input. The `*` multiplies corresponding features: feature 0 in one branch multiplies feature 0 in the other. This is elementwise multiplication, so the feature count stays `d_ff`. Finally, `down_proj(...)` forms learned linear combinations of those products. There is no activation after this last projection.

The word “gate” describes the branch's role, not a restriction on its values. SiLU can be negative and can exceed one. Consequently, it can reverse the sign or enlarge a feature as well as weaken it.

## A fully controlled numerical example

Set the gate projection to the identity, the up projection to twice the identity, and the down projection to produce `[a + b, a - b]` from intermediate features `[a, b]`. This removes random initialization from the calculation.

For the first token `[0, 1]`, the gate becomes `[0, 0.7311]`, the up branch becomes `[0, 2]`, and their product is `[0, 1.4621]`. The output is therefore `[1.4621, -1.4621]`.

For `[1, -1]`, the product is approximately `[1.4621, 0.5379]`. The negative gate and negative up feature multiply into a positive second product. The final output is `[2.0000, 0.9242]`.

| Token | Activated gate | Up branch | Elementwise product | Output |
|---|---|---|---|---|
| `[0, 1]` | `[0, 0.7311]` | `[0, 2]` | `[0, 1.4621]` | `[1.4621, -1.4621]` |
| `[1, -1]` | `[0.7311, -0.2689]` | `[2, -2]` | `[1.4621, 0.5379]` | `[2, 0.9242]` |

Run this after loading the reference implementation:

```python
# Example: Trace two tokens through controlled SwiGLU weights
model = SwiGLUMLP(d_model=2, d_ff=2)
with torch.no_grad():
    model.gate_proj.weight.copy_(torch.eye(2))
    model.gate_proj.bias.zero_()
    model.up_proj.weight.copy_(2 * torch.eye(2))
    model.up_proj.bias.zero_()
    model.down_proj.weight.copy_(torch.tensor([[1., 1.], [1., -1.]]))
    model.down_proj.bias.zero_()
x = torch.tensor([[[0., 1.], [1., -1.]]])
y = model(x)
s = torch.sigmoid(torch.tensor(1.)).item()
expected = torch.tensor([[[2*s, -2*s], [2., 4*s - 2]]])
assert y.shape == (1, 2, 2)
assert torch.allclose(y, expected, atol=1e-6)
y.sum().backward()
assert x.grad is None  # This input did not request gradients.
assert model.gate_proj.weight.grad is not None
assert model.up_proj.weight.grad is not None
assert model.down_proj.weight.grad is not None
```

The final assertions also show that all three projections participate in differentiation. An input needs `requires_grad=True` if you want its own gradient.

## Common mistakes and reference scope

Use `*` between the branches, not `@`. Apply SiLU only to the gate branch, and send the original input to both projections. Preserve the default biases when reproducing this implementation. The reference has no residual connection, normalization, or dropout; those would require additional code around this MLP.
