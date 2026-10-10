# Implement LayerNorm

LayerNorm normalizes one feature vector at a time. For a batch of token representations, each token gets its own mean and variance across features. Nothing in this function mixes one example or token with another.

## Read the shapes before the arithmetic

Let `B` be batch size, `S` token count, and `D` features per token. An input can be `(B, S, D)` or, more generally, `(..., D)`. `gamma` and `beta` have shape `(D,)`. They apply a separate learned scale and shift to each feature. The output keeps the input shape.

For `(2, 3, 4)`, the mean and variance each have shape `(2, 3, 1)`: six independent scalar statistics, one for every token. This size-one last axis is what makes broadcasting work.

## 1. Compute a mean across the last axis

```python
mean = x.mean(dim=-1, keepdim=True)
```

`dim=-1` chooses features. `keepdim=True` keeps that axis at size one. For rows `[1, 3]` and `[2, 6]`, means are `[[2], [4]]`, with shape `(2, 1)`. Subtracting them from the `(2, 2)` input expands each row's single mean across its two features, producing `[[-1, 1], [-2, 2]]`.

The mean is not one number for the whole batch. Reducing over `dim=0` would instead compare examples feature by feature and implement different behavior.

## 2. Use population variance for each vector

```python
var = x.var(dim=-1, keepdim=True, unbiased=False)
```

`unbiased=False` means divide the sum of squared deviations by `D`, not `D-1`. For `[1, 3]`, deviations are `[-1, 1]`, squared deviations are `[1, 1]`, and variance is `(1+1)/2 = 1`. For `[2, 6]`, the same calculation gives `4`. Thus `var` is `[[1], [4]]`.

This convention matters. With two features, dividing by `D-1` would double each variance and produce different normalized outputs.

## 3. Center, rescale, and then apply the affine parameters

```python
x_norm = (x - mean) / torch.sqrt(var + eps)
return gamma * x_norm + beta
```

Subtracting the mean centers each row. Dividing by its standard deviation makes the row's magnitude comparable to other rows. Epsilon goes **inside the square root** and prevents a zero denominator when all features are equal. With default positive epsilon, the variance after normalization is close to one rather than exactly one.

For easy arithmetic, temporarily take `eps=0` on the two nonconstant rows. Both normalize to `[-1, 1]`: the second row's deviations are twice as large, but so is its standard deviation. Choose `gamma=[2, 1]` and `beta=[0.5, -0.5]`; both rows then become `[-1.5, 0.5]`.

| Stage | Row `[1, 3]` | Row `[2, 6]` |
|---|---|---|
| Subtract own mean | `[-1, 1]` | `[-2, 2]` |
| Divide by own standard deviation, with `eps=0` | `[-1, 1]` | `[-1, 1]` |
| Multiply `gamma`, add `beta` | `[-1.5, 0.5]` | `[-1.5, 0.5]` |

The `(D,)` affine parameters broadcast across all leading dimensions. They are supplied by the caller; this function does not create or register them.

```torchcode-visual
normalization
```

## Run the normalization and affine steps

```python
# Example: Rows with different scales normalize the same way
x = torch.tensor([[1., 3.], [2., 6.]], requires_grad=True)
gamma = torch.tensor([2., 1.], requires_grad=True)
beta = torch.tensor([0.5, -0.5], requires_grad=True)
assert torch.equal(x.mean(dim=-1, keepdim=True), torch.tensor([[2.], [4.]]))
assert torch.equal(x.var(dim=-1, keepdim=True, unbiased=False), torch.tensor([[1.], [4.]]))
out = my_layer_norm(x, gamma, beta, eps=0.)
assert torch.allclose(out, torch.tensor([[-1.5, 0.5], [-1.5, 0.5]]))
out.sum().backward()
assert torch.equal(gamma.grad, torch.tensor([-2., 2.]))
assert torch.equal(beta.grad, torch.tensor([2., 2.]))
assert x.grad is not None
constant = torch.tensor([[5., 5.]])
assert torch.equal(my_layer_norm(constant, gamma.detach(), beta.detach()), beta.detach().unsqueeze(0))
```

The constant row has zero centered values, so positive epsilon keeps the result finite and the final answer is just `beta`. Gradients reach `x`, `gamma`, and `beta`; every input coordinate also influences its vector's mean and variance, not merely its own final coordinate.

## Reference scope

This reference normalizes only the last dimension. It does not implement multi-axis `normalized_shape`, construct learned parameters, or manage module state. Its exact variance convention and broadcasting rules are the ones required by this exercise.
