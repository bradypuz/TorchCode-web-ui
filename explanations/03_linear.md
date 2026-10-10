# Simple Linear Layer

## Goal and intuition

A fully connected layer learns a separate weighted combination of input features for each output feature. Its bias shifts each output independently. The exercise builds those trainable tensors directly and uses matrix multiplication rather than `nn.Linear`.

**Shapes:** input `x` has last dimension `in_features`; a typical batch is `(B, in_features)`. `weight` is `(out_features, in_features)`, `bias` is `(out_features,)`, and the output is `(B, out_features)`. PyTorch's matrix multiplication also supports compatible leading batch dimensions, so the same expression works on token features shaped `(B, S, in_features)`.

## Walk through the reference

1. **Initialize the weight.** `torch.randn(out_features, in_features)` gives each output neuron its own row of input coefficients. Multiplication by `1 / sqrt(in_features)` reduces the initialization scale as the number of summed input features grows.
2. **Enable parameter gradients.** The reference calls `weight.requires_grad_(True)` after initialization. Since no gradient-tracked source participated in that initialization, the resulting weight is a leaf tensor whose `.grad` can be populated. The zero bias is also created with `requires_grad=True`.
3. **Compute the affine transformation.** `x @ weight.T` contracts the input feature dimension, producing one value for every output neuron. Adding `bias` broadcasts its output-feature values across the batch or other leading dimensions.

## Why the transpose matters

Each row of `weight` represents one output neuron. For one sample, the computation is

$$
y_j = \sum_{i=1}^{\text{in\_features}} x_i W_{ji} + b_j.
$$

Transposing `(out_features, in_features)` gives the `(in_features, out_features)` matrix required by the right side of `x @ ...`. Autograd differentiates the matrix product and broadcast addition, accumulating parameter gradients across all samples.

## Pitfalls and reference limits

Do not swap the weight dimensions merely to avoid a transpose: the judge inspects the stored shape. The initialization is the specific Gaussian scaling requested by the exercise, not the full default initialization behavior of `nn.Linear`. `SimpleLinear` is a plain Python class rather than an `nn.Module`, so it has no automatic `.parameters()`, `.to()`, or `state_dict()` support; training code must manage these tensors explicitly. The constructor uses the current default device and dtype and assumes a positive input-feature count. Tests verify parameter shapes, the forward formula, and gradient flow.
