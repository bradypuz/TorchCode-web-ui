This function implements **batch normalization**: it normalizes each input feature, then applies a scale (`gamma`) and an offset (`beta`). It behaves differently during **training** and **inference**, keeping running statistics for inference.

It is written primarily for a 2D input with shape:

```python
x.shape == (batch_size, num_features)
```

Each column is a feature, and each row is a sample.

## 1. What the arguments mean

| Argument | Purpose |
|---|---|
| `x` | The input tensor to normalize. |
| `gamma` | A scale applied after normalization, typically a trainable tensor of shape `(num_features,)`. |
| `beta` | An offset applied after normalization, typically a trainable tensor of shape `(num_features,)`. |
| `running_mean` | A tensor storing the moving average of each feature’s mean. Usually initialized to zeros. |
| `running_var` | A tensor storing the moving average of each feature’s variance. Usually initialized to ones. |
| `eps` | A small constant for numerical stability. |
| `momentum` | How much the current batch contributes to the running statistics. |
| `training` | Whether to use training or inference behavior. |

## 2. During training: calculate statistics from the current batch

```python
batch_mean = x.mean(dim=0)
batch_var = x.var(dim=0, unbiased=False)
```

`dim=0` means “reduce across the rows,” so these operations produce one mean and one variance **per feature**.

For example:

```python
x = torch.tensor([
    [1.0, 10.0],
    [3.0, 20.0],
    [5.0, 30.0],
])
```

Then:

```python
batch_mean = tensor([3.0, 20.0])
batch_var  = tensor([2.6667, 66.6667])
```

`unbiased=False` calculates variance by dividing by the batch size \(N\), rather than \(N-1\):

\[
\mu_j = \frac{1}{N}\sum_{i=1}^{N}x_{ij}
\]

\[
\sigma_j^2 = \frac{1}{N}\sum_{i=1}^{N}(x_{ij}-\mu_j)^2
\]

Here, \(j\) identifies a feature and \(i\) identifies a sample.

## 3. Update the running statistics

```python
running_mean.mul_(1 - momentum).add_(momentum * batch_mean.detach())
running_var.mul_(1 - momentum).add_(momentum * batch_var.detach())
```

These implement an **exponential moving average**:

\[
\text{running mean}_{new}
=
(1-m)\,\text{running mean}_{old}
+
m\,\text{batch mean}
\]

\[
\text{running variance}_{new}
=
(1-m)\,\text{running variance}_{old}
+
m\,\text{batch variance}
\]

With the default `momentum=0.1`, the updated statistics are:

- **90%** previous running statistics
- **10%** current batch statistics

Two details matter:

- **`mul_` and `add_` update in place.** The trailing underscore means the original tensor is modified, so the caller’s `running_mean` and `running_var` are updated too.
- **`.detach()` prevents gradients from flowing through these updates.** Running statistics are bookkeeping values, not parameters to optimize.

Next:

```python
mean = batch_mean
var = batch_var
```

So **the current training output uses the current batch’s statistics**, not the newly updated running statistics.

The batch statistics used for normalization are *not* detached, so gradients can still flow through their computation.

## 4. During inference: use the stored statistics

```python
else:
    mean = running_mean
    var = running_var
```

When `training=False`, the function:

- Does not calculate batch statistics.
- Does not update running statistics.
- Normalizes using the statistics accumulated during training.

This means a sample’s normalization does not depend on which other samples happen to be in its inference batch.

## 5. Normalize, then scale and shift

```python
x_norm = (x - mean) / torch.sqrt(var + eps)
```

This performs:

\[
\hat{x}_{ij}
=
\frac{x_{ij}-\mu_j}{\sqrt{\sigma_j^2+\epsilon}}
\]

Subtracting the mean centers each feature around zero. Dividing by the standard deviation gives it approximately unit variance during training.

`eps` prevents division by zero or by an extremely small number.

Finally:

```python
return gamma * x_norm + beta
```

This computes:

\[
y_{ij} = \gamma_j\hat{x}_{ij}+\beta_j
\]

- `gamma` controls the feature’s scale.
- `beta` controls the feature’s offset.

If `gamma=1` and `beta=0`, the output is just the normalized input. If they are trainable tensors included in an optimizer, the model can learn a useful scale and offset instead of being forced to keep zero-mean, unit-variance features.

PyTorch broadcasts the feature-sized tensors across the batch automatically.

## Important caveats

1. **This is not an exact reproduction of `torch.nn.BatchNorm1d`.**  
   PyTorch also uses the biased variance (`unbiased=False`) to normalize a training batch, but updates its running variance using an **unbiased** estimate. This implementation uses the biased variance for both.

2. **The reduction dimensions assume a 2D feature matrix.**  
   For images shaped `(N, C, H, W)`, standard channel-wise batch normalization computes statistics across dimensions `(0, 2, 3)`, not just `0`.

3. **Running statistics should normally have `requires_grad=False`.**  
   They are persistent state, not learnable parameters.

4. **`training=False` does not disable autograd.**  
   It only changes which statistics are used. For inference without gradient tracking, use `torch.no_grad()` or `torch.inference_mode()` separately.

**In short:** during training, the function normalizes using the current batch and updates a running summary; during inference, it normalizes using that stored summary. In both cases, it finishes with `gamma * normalized_input + beta`.