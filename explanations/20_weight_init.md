# Kaiming weight initialization

## What the function does

Fill an existing weight tensor with random values whose spread depends on how many inputs contribute to each output. If a neuron adds many weighted inputs, starting with smaller weights helps keep its outputs from becoming excessively large. Kaiming initialization chooses a scale suited to layers followed by ReLU, which removes much of the negative signal.

`weight` is a floating-point tensor. For a linear layer it has shape `(O, I)`, where `O` is the output width and `I` the input width. The function modifies this tensor in place and returns the very same tensor object. Its shape does not change.

## Choose the number of contributing inputs

```python
fan_in = weight.shape[1] if weight.dim() >= 2 else weight.shape[0]
```

`fan_in` means the number of incoming values combined to produce one output. For a linear weight `(O, I)`, that is `I`, because each output uses one entire row containing `I` weights. `weight.dim()` asks how many axes the tensor has. The fallback treats the length of a 1-D tensor as its incoming count.

This line matters because the layout is output-by-input, not input-by-output. For `(64, 16)`, the correct count for this reference is `16`, not `64`.

```python
std = math.sqrt(2.0 / fan_in)
```

`std` is the standard deviation: roughly how widely sampled values spread around zero. With independent inputs, adding more terms tends to increase the variance of their sum. Dividing by `fan_in` compensates for that growth. The numerator `2` supplies extra variance for the ReLU setting. The square root converts variance into standard deviation.

| Linear weight shape | `fan_in` | Variance `2 / fan_in` | Sampling standard deviation |
|---|---|---|---|
| `(2, 2)` | `2` | `1` | `1` |
| `(2, 8)` | `8` | `0.25` | `0.5` |
| `(64, 16)` | `16` | `0.125` | `0.35355` |
| `(64, 256)` | `256` | `0.0078125` | `0.08839` |

Increasing the incoming width from `16` to `256` increases the count sixteenfold, so the standard deviation decreases fourfold.

## Fill the existing tensor without recording a training operation

```python
with torch.no_grad():
    weight.normal_(0, std)
return weight
```

`.normal_(0, std)` samples from a normal distribution with mean zero and the chosen standard deviation. Its trailing underscore means it changes the tensor's stored values instead of creating a separate result. For initialization, this is what the caller needs: the layer already owns the parameter that should be filled.

`torch.no_grad()` prevents this fill from being recorded in the computation graph. It also permits modifying a trainable parameter directly during initialization. It does not permanently freeze the parameter; subsequent forward passes can still produce gradients for it.

The last line returns the same object for convenience. A caller may write `kaiming_init(layer.weight)` without replacing `layer.weight` or disconnecting it from its module.

## A seeded fill and a scale comparison

For a `(2, 2)` tensor, the scale is one. Seed zero gives approximately `[[1.5410, -0.2934], [-2.1788, 0.5684]]` on CPU. The individual values can be large or small: the prescribed mean and standard deviation describe the sampling distribution, not exact statistics of four samples.

```python
# Example: Fill a trainable matrix and compare incoming-width scales
weight = torch.nn.Parameter(torch.empty(2, 2))
torch.manual_seed(0)
result = kaiming_init(weight)
expected = torch.tensor([[1.5410, -0.2934], [-2.1788, 0.5684]])
assert result is weight
assert torch.allclose(weight, expected, atol=1e-4)
assert weight.requires_grad
assert weight.grad is None

torch.manual_seed(1)
narrow = kaiming_init(torch.empty(1024, 16))
torch.manual_seed(1)
wide = kaiming_init(torch.empty(1024, 256))
assert abs(narrow.std().item() - math.sqrt(2 / 16)) < 0.01
assert abs(wide.std().item() - math.sqrt(2 / 256)) < 0.003
assert narrow.std() > wide.std()
```

The larger tensors make the empirical standard deviations closer to their requested values. Setting a seed makes this demonstration reproducible, but different seeds are valid initializations too.

## Common mistakes and reference scope

Initialize weights before training; calling this during training discards their learned values. This reference's incoming-count rule is correct for ordinary 2-D linear weights. For a convolutional tensor `(O, I, kH, kW)`, it uses only `I` and omits the kernel area, so it is not a general replacement for PyTorch's convolution-aware Kaiming initializer. Use nonempty tensors with a positive incoming count.
