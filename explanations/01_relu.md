# Implement ReLU

## Goal and intuition

ReLU keeps positive activations and replaces non-positive activations with zero. It introduces a simple nonlinearity: a network built only from linear layers would still be one linear transformation, while ReLU lets different inputs activate different paths. This exercise implements the operation directly, without a built-in activation or clamp.

**Input and output:** `x` is a floating-point tensor of any shape. The output has the same shape and device. For example, `[-2, 0, 3]` becomes `[0, 0, 3]` for ordinary finite inputs.

## Walk through the reference

1. **Find positive elements.** `x > 0` creates a Boolean tensor with `True` wherever an input is strictly positive. The comparison preserves every dimension of `x`.
2. **Convert the mask into numbers.** `.float()` maps `True` to `1.0` and `False` to `0.0`, using float32. This produces a multiplier for each element.
3. **Apply the gate.** Multiplying `x` by that mask keeps positive values and zeros finite negative values. The entire operation is vectorized, so no Python loop is needed.

## Why gradients still work

The comparison does not carry a useful derivative, but the multiplication still depends on `x`. Autograd treats the mask as a constant multiplier. Consequently, the derivative is one on positive inputs and zero on negative inputs:

$$
\frac{\partial y_i}{\partial x_i} = \mathbf{1}[x_i > 0].
$$

At zero, ReLU is not differentiable in the ordinary sense. This implementation chooses a zero gradient because its condition uses strict `>`; the judge permits either zero or one at that point.

## Pitfalls and reference limits

Avoid reconstructing the result with `torch.tensor(...)` or converting through NumPy: that would disconnect the gradient path. The mask multiplication also has two concrete limitations. A float32 mask can promote lower-precision inputs, so output dtype preservation is not guaranteed. Also, `-inf * 0` is `NaN`, whereas mathematical ReLU of negative infinity is zero. The provided tests exercise finite float32 values, shapes, gradients, and vectorized performance. Treat this as their compact reference implementation, rather than an exact replacement for every built-in ReLU edge case.
