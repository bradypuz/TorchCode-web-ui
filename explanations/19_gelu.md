# GELU activation

## What this function does

GELU changes each input number independently using a smooth curve. Large positive values mostly pass through. Large negative values get strongly reduced, while small negative values can remain negative. This makes it different from ReLU, which sets every negative number to exactly zero.

The input `x` is a floating-point tensor of any shape, and the output has the same shape. There are no learned parameters, no batch statistics, and no interaction between positions. If `x` has shape `(B, S, D)`, where `B` is the batch size, `S` the sequence length, and `D` the feature count, the function transforms all `B * S * D` numbers separately.

## Read the exact expression

```python
return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
```

Start with `x / math.sqrt(2.0)`. This rescales each number for the error function, `erf`. The error function is a smooth mathematical function ranging from approximately `-1` to `1`. At zero it is zero; positive inputs give positive results and negative inputs give negative results.

Next, `1.0 + torch.erf(...)` shifts that range to approximately zero through two. Multiplication by `0.5` makes a factor between zero and one. Finally, multiplication by `x` scales the original input using this factor. Scalars such as `0.5` apply to every tensor entry automatically.

The combined factor `0.5 * (1 + erf(x / sqrt(2)))` is also the probability that a standard normal random number is less than `x`. You do not need to draw any random numbers to compute it: `torch.erf` evaluates the function directly. GELU itself is deterministic.

For `x = 1`, the factor is about `0.8413`, so the result is `0.8413`. For `x = -1`, it is about `0.1587`, giving `-0.1587`. At zero the factor is one half, but multiplying by zero still returns zero.

## See the curve through concrete values

| Input `x` | Scaling factor | GELU output | ReLU output |
|---|---|---|---|
| `-2` | `0.02275` | `-0.04550` | `0` |
| `-1` | `0.15866` | `-0.15866` | `0` |
| `0` | `0.50000` | `0` | `0` |
| `1` | `0.84134` | `0.84134` | `1` |
| `2` | `0.97725` | `1.95450` | `2` |

Notice that `-2` produces a result closer to zero than `-1`. The negative part of the curve is not simply a straight line with a small constant slope. On the positive side, the factor approaches one as inputs grow, so GELU gradually approaches the identity operation.

```python
# Example: Evaluate the exact GELU curve at five familiar points
x = torch.tensor([[-2., -1., 0., 1., 2.]], requires_grad=True)
y = my_gelu(x)
expected = torch.tensor([[-0.0455003, -0.1586553, 0., 0.8413447, 1.9544997]])
assert y.shape == (1, 5)
assert torch.allclose(y, expected, atol=1e-6)
y.sum().backward()
assert torch.allclose(x.grad[0, 2], torch.tensor(0.5), atol=1e-6)
assert torch.isfinite(x.grad).all()
```

The gradient at zero is `0.5`: very small input changes near zero produce roughly half as much output change. PyTorch differentiates the error-function expression automatically.

## Common mistakes and reference scope

Use the original `x` in the final multiplication. Removing it would return only a factor between zero and one. This is the exact error-function formulation of GELU; the frequently used formula involving `tanh` and a cubic term is an approximation and can differ slightly. The reference preserves tensor shape and remains differentiable, but expects a dtype and device supported by `torch.erf`.
