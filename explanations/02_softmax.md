# Implement Softmax

## Goal and intuition

Softmax turns a vector of scores into non-negative weights that sum to one. Larger scores receive larger weights, but the result depends on relative score differences rather than their absolute offset. The reference implements softmax along a chosen dimension using tensor reductions and exponentiation.

**Input and output:** `x` can have any nonempty shape, and `dim` selects the axis to normalize; it defaults to the last axis. The output retains the input shape. For `(B, C)` with `dim=-1`, every batch row is normalized independently across its `C` entries.

## Walk through the reference

1. **Find the largest score on each slice.** `x.max(dim=dim, keepdim=True).values` keeps the reduced axis at size one. A `(B, C)` input therefore produces `(B, 1)` maxima that broadcast correctly across their rows.
2. **Exponentiate shifted scores.** The code computes `exp(x - x_max)`. Every finite shifted score is at most zero, so every exponential is at most one. This prevents the overflow caused by directly exponentiating values such as `1000`.
3. **Normalize along the same axis.** Summing exponentials with `keepdim=True` produces one denominator per slice. Division broadcasts that denominator back to every element.

## Why subtracting the maximum is valid

Subtracting a shared constant cancels between numerator and denominator:

$$
\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}
= \frac{e^{x_i}/e^m}{\sum_j e^{x_j}/e^m}
= \frac{e^{x_i}}{\sum_j e^{x_j}}.
$$

For a finite nonempty slice, at least one shifted value is zero, so at least one exponential is exactly one. The denominator cannot vanish solely because smaller exponentials underflow. All operations remain connected to autograd.

## Pitfalls and reference limits

The reduction and division must use the same `dim`; omitting `keepdim=True` often breaks broadcasting. Numerical stability here concerns finite scores, not arbitrary special values: a slice containing positive infinity, or entirely negative infinity, can produce `NaN` through infinity subtraction. Empty reduction axes are unsupported, and the code does not upcast low-precision tensors for accumulation. The judge compares ordinary vectors and matrices with PyTorch softmax and explicitly checks large finite logits. These cases demonstrate why the maximum subtraction belongs before exponentiation.
