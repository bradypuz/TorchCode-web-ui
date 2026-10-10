# Implement LayerNorm

## Goal and intuition

Layer normalization rescales each feature vector independently. For a sequence of token embeddings, every token computes its own mean and variance across its features, without depending on other tokens or batch examples. Learnable scale and shift then let the model choose how those normalized features should be represented.

**Shapes:** `x` is `(..., D)`, while `gamma` and `beta` are `(D,)`. The output is `(..., D)`. Intermediate means and variances are `(..., 1)`, which broadcast across the last feature dimension.

## Walk through the reference

1. **Calculate the mean.** `x.mean(dim=-1, keepdim=True)` computes one mean for each feature vector. Keeping the last dimension allows direct subtraction from `x`.
2. **Calculate the population variance.** `x.var(..., unbiased=False)` divides the squared deviations by `D`, rather than `D-1`. This is the variance convention required for the exercise and for the corresponding single-axis LayerNorm comparison.
3. **Standardize and apply the affine parameters.** `(x - mean) / sqrt(var + eps)` centers and rescales the features. Multiplying by `gamma` and adding `beta` applies one learned scale and shift per feature.

## Why it works

For each feature vector,

$$
\mu = \frac{1}{D}\sum_i x_i, \qquad
\sigma^2 = \frac{1}{D}\sum_i (x_i-\mu)^2,
\qquad y_i = \gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i.
$$

The centered values sum to approximately zero. With unit scale and zero shift, their variance is close to one when the original variance is large relative to `eps`. Adding epsilon inside the square root prevents division by zero for a constant vector. It deliberately makes the normalized variance slightly smaller than one; exact unit variance is not the contract.

## Pitfalls and reference limits

Using the default unbiased variance changes the answer, especially for small `D`. Normalizing over `dim=0` instead would mix examples and implement a different operation. All arithmetic stays in the autograd graph, so gradients can reach the input, scale, and shift. This reference normalizes only the last dimension; it does not implement the arbitrary multi-axis `normalized_shape` option of `nn.LayerNorm`. It also does not create or register the affine parameters, which must be supplied by the caller. The judge checks two- and three-dimensional inputs, nontrivial affine parameters, and gradient connectivity.
