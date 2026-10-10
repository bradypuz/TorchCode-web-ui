# LoRA linear layer

## What the layer does

Keep an existing linear transformation fixed while learning an additional small update through two narrower matrices. This is the core LoRA idea: train a low-rank change to a layer instead of changing its full weight matrix.

`in_features = I` is the input width, `out_features = O` the output width, and `rank = R` the intermediate adapter width. An input of shape `(..., I)` produces `(..., O)`. The dots represent any leading dimensions, such as batch and sequence. `alpha` controls the update's scale; this reference sets `scaling = alpha / R` and therefore needs a positive rank.

## Freeze the base transformation

```python
self.linear = nn.Linear(in_features, out_features)
self.linear.weight.requires_grad_(False)
self.linear.bias.requires_grad_(False)
```

After `super().__init__()`, the constructor creates a standard linear layer with weight shape `(O, I)` and bias shape `(O,)`. Setting `requires_grad` to false means backpropagation does not accumulate gradients into these parameters. It does not stop gradients from passing through the layer to a differentiable input.

This exercise creates the base layer from scratch. In a fine-tuning application, you would first load the pretrained base weights into it; freezing freshly initialized weights alone does not produce a pretrained model.

## Create a narrow trainable path

```python
self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
self.scaling = alpha / rank
```

`lora_A` has shape `(R, I)` and starts with small random numbers. `lora_B` has shape `(O, R)` and starts at zero. Both are registered trainable parameters. Together they use `R * I + O * R` numbers, compared with `O * I` numbers in the full base weight. This is a parameter reduction when `R` is sufficiently small relative to both widths.

Zero initialization of `B` makes the complete adapter output zero at the start even though `A` is nonzero. Consequently, the initial layer behaves exactly like the base transformation. Starting both matrices at zero would also make both first-step gradients zero, blocking the adapter from starting to learn.

## Follow the two matrix multiplications

```python
return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
```

The base branch computes `x @ linear.weight.T + linear.bias`. In the adapter branch, `A.T` has shape `(I, R)`, so the first multiplication maps each input down to `R` features. `B.T` has shape `(R, O)`, so the second maps it to the output width. Multiplication by `alpha / R` controls the size of the addition.

```text
                         fixed weight and bias
x (..., I) ───────────────→ base (..., O) ──────────┐
  │                                               + → output (..., O)
  └→ A.T (I, R) → (..., R) → B.T (R, O) → scale ──┘
```

The equivalent weight update is `scaling * (B @ A)`, shaped `(O, I)`. The product can only form combinations through `R` intermediate features, explaining the term “low rank.” The forward pass need not construct that full matrix.

## A rank-one update with explicit numbers

Let the base be the two-dimensional identity, with bias `[1, -1]`, and let `x = [2, 3]`. Its output is `[3, 2]`. Set `A = [[1, 2]]`, `B = [[3], [4]]`, rank one, and `alpha = 0.5`.

| Stage | Calculation | Result |
|---|---|---|
| Base | `[2, 3] + [1, -1]` | `[3, 2]` |
| Narrow feature | `2*1 + 3*2` | `[8]` |
| Expanded update | `[8] @ [[3, 4]]` | `[24, 32]` |
| Scaled update | `[24, 32] * 0.5` | `[12, 16]` |
| Final output | `[3, 2] + [12, 16]` | `[15, 18]` |

```python
# Example: Add a rank-one correction while keeping base parameters frozen
layer = LoRALinear(in_features=2, out_features=2, rank=1, alpha=0.5)
with torch.no_grad():
    layer.linear.weight.copy_(torch.eye(2))
    layer.linear.bias.copy_(torch.tensor([1., -1.]))
    layer.lora_A.copy_(torch.tensor([[1., 2.]]))
x = torch.tensor([[2., 3.]], requires_grad=True)
assert torch.equal(layer(x), torch.tensor([[3., 2.]]))  # B is initially zero.
layer(x).sum().backward()
assert torch.equal(layer.lora_A.grad, torch.zeros_like(layer.lora_A))
assert torch.equal(layer.lora_B.grad, torch.tensor([[4.], [4.]]))
assert layer.linear.weight.grad is None
assert layer.linear.bias.grad is None
with torch.no_grad():
    layer.lora_B.copy_(torch.tensor([[3.], [4.]]))
assert torch.equal(layer(x), torch.tensor([[15., 18.]]))
assert layer(x).shape == (1, 2)
```

The first backward pass gives `B` a gradient but gives `A` a zero gradient because its path passes through zero `B`. Once `B` becomes nonzero, later gradients can train `A` too. Frozen base parameters remain unchanged by gradient-based training.

## Common mistakes and reference scope

Transpose the adapter matrices according to their stored shapes and scale only the adapter branch. Freezing the base is distinct from initializing it with pretrained values. This reference has no adapter dropout, weight-merging method, or rank validation, and it freezes the base bias as well as its weight.
