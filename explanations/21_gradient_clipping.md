# Gradient norm clipping

## What the function does

After backpropagation, limit the combined size of a model's gradients before the optimizer uses them. The function scales all participating gradients by the same factor. This preserves their relative directions while preventing one step from using an excessively large gradient vector.

`parameters` is an iterable of parameter tensors, usually `model.parameters()`. Each parameter may have a `.grad` tensor of the same shape or `.grad = None`. `max_norm` is the allowed combined Euclidean norm: square every gradient entry, add the squares, and take the square root. The function changes `.grad` tensors in place and returns a Python float containing the **original** norm.

## Keep only parameters with gradients

```python
parameters = [p for p in parameters if p.grad is not None]
```

Some parameters may be unused in a particular forward pass, or frozen, and therefore lack a gradient. Skipping them prevents attempts to call methods on `None`. Making a list also lets the function traverse the same parameters twice; an incoming generator could otherwise be exhausted after the norm calculation.

## Treat every gradient as one large vector

```python
total_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in parameters))
```

`p.grad.norm()` gives the Euclidean norm of one parameter's gradient. Squaring that norm gives the sum of squared entries for that parameter. Adding these squared norms and taking the square root is equivalent to concatenating all gradient entries into one long vector and measuring its norm.

For two gradient tensors `[3, 4]` and `[12]`, their individual norms are `5` and `12`. The combined norm is `sqrt(5**2 + 12**2) = 13`. Adding the individual norms would incorrectly give `17`.

## Calculate one scale and apply it only when needed

```python
clip_coef = max_norm / (total_norm + 1e-6)
if clip_coef < 1:
    for p in parameters:
        p.grad.mul_(clip_coef)
return total_norm.item()
```

If the norm is `13` and the requested limit is `6.5`, the coefficient is approximately `0.5`. Every gradient is multiplied by that coefficient, bringing the combined norm just below `6.5`. The small `1e-6` protects division when the norm is zero and means the result is slightly below the exact boundary.

When the norm is already below the limit, the coefficient exceeds one. The `if` prevents increasing those already-small gradients. `.mul_()` changes the existing gradient tensor, so the optimizer later reads the clipped values. `.item()` converts the original scalar norm tensor to a Python number; it does not recompute the norm after scaling.

## Follow a complete example

| Gradient tensor | Before clipping | After clipping to approximately half |
|---|---|---|
| First parameter | `[3, 4]` | `[1.5, 2]` |
| Second parameter | `[12]` | `[6]` |
| Combined norm | `13` | approximately `6.5` |

```python
# Example: Clip a 3-4-12 gradient vector with one shared scale
first = torch.nn.Parameter(torch.zeros(2))
second = torch.nn.Parameter(torch.zeros(1))
unused = torch.nn.Parameter(torch.zeros(1))
first.grad = torch.tensor([3., 4.])
second.grad = torch.tensor([12.])
original_norm = clip_grad_norm([first, second, unused], max_norm=6.5)
scale = 6.5 / (13 + 1e-6)
assert original_norm == 13.0
assert torch.allclose(first.grad, torch.tensor([3., 4.]) * scale)
assert torch.allclose(second.grad, torch.tensor([12.]) * scale)
combined = torch.sqrt(first.grad.square().sum() + second.grad.square().sum())
assert abs(combined.item() - 6.5) < 1e-5
assert unused.grad is None
before = first.grad.clone()
clip_grad_norm([first, second], max_norm=20)
assert torch.equal(first.grad, before)
```

The second call demonstrates the below-limit case: it returns the current norm but leaves the gradients unchanged. In a training loop, the intended order is `loss.backward()`, then clipping, then `optimizer.step()`.

## Common mistakes and reference scope

Clip gradients, not parameter values. Do not normalize each tensor separately; that changes their relative sizes. This reference assumes a nonempty list of dense gradients on compatible devices. If every `.grad` is `None`, Python's empty sum becomes an integer and `torch.sqrt` fails. It has no nonfinite-gradient handling and expects a nonnegative norm limit.
