# Implement BatchNorm

## Goal and intuition

Batch normalization standardizes each feature using statistics across examples. Training uses the current batch, while inference uses running statistics collected from earlier batches. This exercise makes that state change explicit rather than hiding it inside a module.

**Shapes:** `x` is `(N, D)`. `gamma`, `beta`, `running_mean`, and `running_var` are `(D,)`. The output remains `(N, D)`. Running statistics are mutable buffers provided by the caller and should not require gradients.

## Walk through the reference

1. **Choose statistics based on mode.** In training, `mean(dim=0)` and `var(dim=0, unbiased=False)` calculate one mean and population variance per feature. In inference, the code directly selects the supplied running tensors instead.
2. **Update training buffers in place.** `mul_(1 - momentum).add_(momentum * batch_stat.detach())` blends the old estimate with the new batch statistic. In-place operations preserve the caller's buffer identity. Detaching the batch statistics prevents this state update from recording a gradient dependency on the current batch.
3. **Normalize and transform.** The code subtracts the selected mean, divides by `sqrt(var + eps)`, and applies featurewise `gamma` and `beta`. Training normalization uses the original differentiable batch statistics, not their detached versions.

## Why it works

For feature `j`, training computes

$$
y_{ij} = \gamma_j \frac{x_{ij}-\mu_{B,j}}{\sqrt{\sigma^2_{B,j}+\epsilon}}+\beta_j,
\qquad r_{\mathrm{new}}=(1-m)r_{\mathrm{old}}+m r_B.
$$

With unit scale and zero shift, training outputs are approximately centered per feature. During inference, outputs do not depend on other examples in the current batch because their statistics come from the running buffers. Here, larger momentum means the newest batch contributes more, which differs from some optimizer uses of the same word.

## Pitfalls and reference limits

Reassigning a local `running_mean` variable would not update the caller's tensor; the underscore operations matter. Normalizing across the last axis would implement LayerNorm instead. The exercise explicitly updates running variance with the **biased population variance**. PyTorch BatchNorm uses an unbiased estimate for its running variance, so this reference follows the judge's stated contract rather than reproducing every `nn.BatchNorm1d` detail. It supports the two-dimensional feature-batch contract, has no buffer initialization or mode-switching module, and assumes valid positive running variances and non-gradient buffers. Tests cover both modes, state updates, and input/affine gradients.
