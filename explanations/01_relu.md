# Implement ReLU

This solution is a gate applied separately to every tensor element: a positive value passes through, while a finite non-positive value becomes zero. Read the single return statement from the inside out. There are no learned parameters and no dimensions to reduce.

## The input and the promise

`x` is a floating-point tensor. It can be a vector, an image-like tensor, or a batch of token features. The output keeps the same shape and device. If a batch has shape `(B, D)`, `B` means the number of examples and `D` means features per example; the function never mixes examples or features.

The mathematical rule is `max(0, x)` for each coordinate. The exercise bans the built-in ReLU and clamp operations, so the reference expresses that rule using a comparison and multiplication.

## Read the actual statement in three parts

```python
return x * (x > 0).float()
```

**First, `x > 0` asks a yes/no question at every location.** For `x = [-2, -1, 0, 1, 2]`, the answer is `[False, False, False, True, True]`. Notice the strict greater-than condition: zero is in the blocked group. Comparing a tensor with scalar zero automatically applies that same threshold to all coordinates.

**Next, `.float()` turns answers into multipliers.** The Boolean mask becomes a float32 tensor containing `[0, 0, 0, 1, 1]`. Its shape is exactly `x.shape`. A one means “keep this value”; a zero means “remove it.”

**Finally, `x * mask` is elementwise multiplication.** It is not matrix multiplication. Each input is multiplied only by the mask value at the same location, so there is no Python loop and no change in shape.

| Input coordinate | `x > 0` | Float mask | Returned coordinate |
|---:|:---:|---:|---:|
| -2 | False | 0 | 0 |
| 0 | False | 0 | 0 |
| 3 | True | 1 | 3 |

This table also explains why simply returning the mask would be wrong: an input of `3` must return `3`, not `1`.

## What happens during backpropagation?

The comparison itself has no differentiable transition. However, the final multiplication still contains the original `x`, so autograd can follow that path. For the mask produced by this comparison, the derivative is its multiplier: zero on blocked values and one on positive values. At zero, where mathematical ReLU has a corner, this reference chooses a zero gradient.

```python
# Example: Check the gate and its backward gradient
x = torch.tensor([-2., -1., 0., 1., 2.], requires_grad=True)
out = relu(x)
assert out.shape == x.shape
assert torch.equal(out, torch.tensor([0., 0., 0., 1., 2.]))
out.sum().backward()
assert torch.equal(x.grad, torch.tensor([0., 0., 0., 1., 1.]))
matrix = torch.tensor([[-3., 2.], [4., -1.]])
assert torch.equal(relu(matrix), torch.tensor([[0., 2.], [4., 0.]]))
```

Summing the output asks how much that total changes when each input increases slightly. The positive coordinates contribute directly, while the blocked coordinates contribute nothing. Rebuilding the output through NumPy or a new `torch.tensor(...)` would lose that gradient connection.

## Reference scope

The example uses finite float32 values, as the judge does. The float32 mask can promote lower-precision inputs. Also, negative infinity times zero produces `NaN`, so the multiplication trick does not reproduce mathematical ReLU for that special value. These are limits of this exact reference expression, not extra behavior implemented by the code.
