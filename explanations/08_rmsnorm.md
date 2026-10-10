# Implement RMSNorm

## Goal and intuition

RMSNorm rescales each feature vector by its root mean square magnitude. It does not subtract the mean, so it preserves the vector's direction before applying learned featurewise weights. Compared with LayerNorm, the central operation is simpler: measure overall size, divide by it, then scale individual features.

**Shapes:** `x` is `(..., D)` and `weight` is `(D,)`. The root mean square has shape `(..., 1)`, and the output has the same shape as `x`. This works equally for a matrix of examples or a `(B, S, D)` tensor of token representations.

## Walk through the reference

1. **Measure squared magnitude.** `x.pow(2).mean(dim=-1, keepdim=True)` averages squared feature values within each vector. Squaring makes both positive and negative coordinates contribute to the size estimate.
2. **Add stability and take the root.** `sqrt(mean_square + eps)` produces the denominator. Epsilon is placed inside the square root, matching the problem definition and avoiding a zero denominator for a zero vector.
3. **Rescale and apply weights.** `x / rms` divides all coordinates in a feature vector by the same scalar. Multiplication by `weight` then broadcasts a learned scale across leading dimensions.

## Why it works

For one vector,

$$
r=\sqrt{\frac{1}{D}\sum_i x_i^2+\epsilon},
\qquad y_i=\frac{x_i}{r}w_i.
$$

If all weights equal one and epsilon is negligible, the normalized vector's mean square is near one. Its mean need not be zero: a constant nonzero vector remains a constant nonzero vector after normalization. This distinguishes RMSNorm from LayerNorm and is a useful way to check an implementation. All reductions and arithmetic preserve the autograd path to both `x` and `weight`.

## Pitfalls and reference limits

Using the mean of `x` instead of the mean of `x²` measures signed average rather than magnitude. Computing a norm across batch or sequence dimensions also mixes independent vectors. The reference does not supply a bias or center the input, and it does not create trainable weights itself. It performs squaring and reduction in the input dtype; extremely large values can overflow, especially in low precision, because there is no explicit float32 accumulation step. The judge uses finite inputs to check two- and three-dimensional shapes, nontrivial learned scales, agreement with the formula, and gradient flow.
