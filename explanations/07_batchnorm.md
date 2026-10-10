# Implement BatchNorm

This function has two responsibilities: normalize the current input and maintain running statistics for later inference. Those responsibilities use related numbers, but they are deliberately separate. Training normalizes with the **current batch**; inference normalizes with the **stored running estimates**.

## Understand the axes and state

`N` is the number of examples and `D` the number of features. Input `x` and output have shape `(N,D)`. Each of `gamma`, `beta`, `running_mean`, and `running_var` has shape `(D,)`.

Every feature column is one group to normalize. For input rows `[1,10]` and `[3,14]`, the first feature group is `[1,3]` and the second is `[10,14]`. Means are `[2,12]`; population variances are `[1,4]`. A `(D,)` statistic broadcasts down the batch, applying the first statistic to every first feature and the second to every second feature.

## 1. In training, calculate current batch statistics

```python
if training:
    batch_mean = x.mean(dim=0)
    batch_var = x.var(dim=0, unbiased=False)
```

`dim=0` is the example axis. Reducing it leaves one number per feature, shape `(D,)`. `unbiased=False` divides squared deviations by `N`. Our first feature has variance `((-1)^2 + 1^2)/2 = 1`; the second has variance `((-2)^2 + 2^2)/2 = 4`.

These are still differentiable functions of `x`. They will be used directly when normalizing this training batch.

## 2. Update the caller's running buffers in place

```python
running_mean.mul_(1 - momentum).add_(momentum * batch_mean.detach())
running_var.mul_(1 - momentum).add_(momentum * batch_var.detach())
```

A trailing underscore marks an in-place operation. The code first multiplies the old buffer by `1-momentum`, then adds the new statistic times `momentum`. It changes the original tensors the caller supplied; assigning a new local tensor would not have the same stateful effect.

With `momentum=0.5`, initial running mean `[0,0]` becomes `[1,6]`. Initial running variance `[1,1]` becomes `[1,2.5]`. Each new buffer is halfway between its old value and the observed batch statistic. Larger momentum gives more weight to the newest batch.

`.detach()` prevents the **buffer update** from recording a dependency on the batch's computation graph. This does not detach the statistics used for training normalization; the original `batch_mean` and `batch_var` still carry gradients.

## 3. Select the correct statistics for the output

```python
mean = batch_mean
var = batch_var
```

These two assignments are inside the training branch. Updating running statistics does not mean training suddenly uses those blended estimates.

```python
else:
    mean = running_mean
    var = running_var
```

Inference skips current-batch statistics and skips the updates entirely. Its answer for one example therefore does not depend on which other examples happen to be in the inference batch.

## 4. Normalize and apply affine parameters

```python
x_norm = (x - mean) / torch.sqrt(var + eps)
return gamma * x_norm + beta
```

The selected `(D,)` mean, variance, scale, and shift broadcast across all `N` rows. With unit scale, zero shift, and `eps=0` for simple nonconstant arithmetic, our training rows become `[-1,-1]` and `[1,1]`.

| Quantity | Feature 0 | Feature 1 |
|---|---:|---:|
| Training batch mean | 2 | 12 |
| Training batch variance | 1 | 4 |
| Updated running mean at momentum 0.5 | 1 | 6 |
| Updated running variance at momentum 0.5 | 1 | 2.5 |
| Inference output for input `[1,10]`, `eps=0` | 0 | `4/sqrt(2.5)` |

The different inference answer is expected: its statistics represent a running estimate, not the current batch's exact distribution. Positive default epsilon prevents division by zero for constant feature columns.

```torchcode-visual
normalization
```

## Run training, state update, and inference

```python
# Example: Training updates buffers that inference later reads
x = torch.tensor([[1., 10.], [3., 14.]], requires_grad=True)
gamma = torch.ones(2, requires_grad=True)
beta = torch.zeros(2, requires_grad=True)
running_mean = torch.zeros(2)
running_var = torch.ones(2)
training_out = my_batch_norm(x, gamma, beta, running_mean, running_var,
                             eps=0., momentum=0.5, training=True)
assert torch.allclose(training_out, torch.tensor([[-1., -1.], [1., 1.]]))
assert torch.equal(running_mean, torch.tensor([1., 6.]))
assert torch.equal(running_var, torch.tensor([1., 2.5]))
mean_before, var_before = running_mean.clone(), running_var.clone()
inference_out = my_batch_norm(x, gamma, beta, running_mean, running_var,
                             eps=0., training=False)
expected = torch.tensor([[0., 4. / (2.5 ** 0.5)], [2., 8. / (2.5 ** 0.5)]])
assert torch.allclose(inference_out, expected)
assert torch.equal(running_mean, mean_before) and torch.equal(running_var, var_before)
training_out.sum().backward()
assert x.grad is not None and gamma.grad is not None
assert torch.equal(beta.grad, torch.tensor([2., 2.]))
assert not running_mean.requires_grad and not running_var.requires_grad
```

## Reference scope

The caller supplies non-gradient buffers and controls the `training` argument. This exercise tracks **biased population variance** in its running buffer; full PyTorch BatchNorm uses an unbiased estimate for the running variance. The reference targets two-dimensional batches and has no buffer initialization, automatic mode switching, or input validation.
