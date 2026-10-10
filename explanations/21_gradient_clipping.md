# Gradient Norm Clipping

## Goal and intuition

Limit the size of an entire gradient update without changing its direction. Imagine all parameter gradients concatenated into one long vector. If that vector is too long, multiply every component by the same factor. This differs from independently clipping each element or clipping each parameter tensor to its own norm limit.

## Inputs and outputs

- `parameters`: an iterable of tensors or model parameters, after backpropagation has populated their `.grad` fields.
- `max_norm`: the desired global L2 norm bound, normally nonnegative.
- Return value: a Python float containing the **original**, unclipped global norm.

Parameter values are unchanged. Existing gradients are modified in place, so the function is normally used between `loss.backward()` and `optimizer.step()`.

## Walk through the reference

1. Convert the iterable to a list containing only parameters whose `.grad` is not `None`.
2. Compute each gradient tensor's L2 norm, square those norms, add them, and take the square root.
3. Compute `clip_coef = max_norm / (total_norm + 1e-6)`. The small denominator offset prevents division by zero when populated gradients are all zero.
4. If the coefficient is less than one, multiply every gradient by it in place. Otherwise leave the gradients untouched.
5. Return `total_norm.item()` from before the mutation.

## Why it works

$$
G=\sqrt{\sum_p\|g_p\|_2^2},\qquad g'_p=cg_p
$$

$$
c=\min\left(1,\frac{\mathrm{max\_norm}}{G+10^{-6}}\right)
$$

All gradients share the same scaling factor. Thus their relative proportions and the direction of the combined vector are preserved. When clipping occurs, the resulting norm is slightly below the requested bound because of the epsilon.

## Pitfalls and reference scope

Do not return the clipped norm, and do not scale gradients upward when they are already small. The reference has an **empty-list edge case**: if no parameter has a gradient, Python's `sum` returns integer zero and `torch.sqrt` raises a type error. The epsilon does not fix that case. It does not validate negative bounds, handle nonfinite gradients explicitly, or combine gradients across different devices. It computes only L2 clipping, with no configurable norm order. The judge exercises clipping, original-norm return, small-gradient identity, and direction preservation.
