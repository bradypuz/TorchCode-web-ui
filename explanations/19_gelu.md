# GELU Activation

## Goal and intuition

Implement the exact Gaussian Error Linear Unit as an elementwise activation. GELU smoothly scales an input according to the standard normal cumulative distribution function. Large positive values mostly pass through; large negative values are pushed toward zero. Unlike ReLU, small negative values can still produce a negative output.

## Inputs and outputs

- `x`: a floating tensor of any shape.
- Output: a floating tensor with the same shape.

No parameters, reductions, or changes to the feature dimensions are involved. Every tensor element is transformed independently, so the same implementation works for a vector, a matrix, or a batch of token features.

## Walk through the reference

1. Divide `x` by the scalar `math.sqrt(2.0)`.
2. Apply `torch.erf`, the Gaussian error function, elementwise.
3. Add one and multiply by `0.5` to obtain the standard normal cumulative probability at `x`.
4. Multiply that probability by the original `x` and return the result.

## Why it works

The standard normal cumulative distribution function has this error-function form:

$$
\Phi(x)=\frac{1}{2}\left(1+\operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right)
$$

Therefore:

$$
\operatorname{GELU}(x)=x\Phi(x)
$$

At zero, the probability factor is one half, but multiplication by zero makes the output exactly zero. As `x` becomes strongly positive, the factor approaches one; as it becomes strongly negative, the factor approaches zero. These are continuous transitions instead of a hard threshold. PyTorch can differentiate the tensor operations, so gradients propagate through the activation without a custom backward function.

## Pitfalls and reference scope

Keep the original `x` in the final multiplication; returning only the CDF would be a different activation. The denominator is `sqrt(2)`, not `sqrt(2*pi)`. This reference uses the **exact error-function expression**, rather than GELU's common tanh approximation. The task's template requests the exact version even though the judge hint also mentions the approximation. Numerical behavior still depends on the input dtype and ordinary floating-point precision. The reference provides no approximation switch or in-place variant. The judge compares against default `F.gelu`, checks the zero value and shape preservation, and verifies gradient flow.
