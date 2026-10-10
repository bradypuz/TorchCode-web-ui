# INT8 Quantized Linear Layer

`Int8Linear` stores approximate weights as small signed integers. A scale for each output row turns those integers back into floating-point weights during forward. This reduces weight storage while keeping the familiar linear operation `x @ weight.T + bias`.

## Inputs and stored shapes

The constructor takes `weight` of shape `(O, I)` and optional `bias` of shape `(O,)`. Integer weights keep shape `(O, I)`, and scales have shape `(O, 1)`. Input `x` can have shape `(..., I)`; the output has shape `(..., O)`. The weights are quantized once when the layer is constructed.

## Reading the actual code

```python
super().__init__()
scale = weight.abs().amax(dim=1, keepdim=True) / 127.0
```

Calling the parent initializer enables normal module behavior such as buffer registration. `abs()` measures magnitude, and `amax(dim=1)` finds the largest magnitude within each output row. Dividing by `127` maps that largest weight near the largest positive signed INT8 value. `keepdim=True` is essential: a scale shaped `(O, 1)` broadcasts across the input columns in its own row.

For rows `[1, 0.4, -0.2]` and `[10, 4, -2]`, scales are `1/127` and `10/127`. The second row has a ten-times-larger quantization step, but the same relative integer representation.

```python
self.register_buffer('weight_int8',
    torch.round(weight / (scale + 1e-10)).clamp(-128, 127).to(torch.int8))
self.register_buffer('scale', scale)
```

Dividing each row by its scale expresses weights in integer-sized units. Rounding picks the nearest integer; clamping enforces the legal signed INT8 range; `.to(torch.int8)` changes storage dtype. PyTorch's `round` uses nearest-even behavior for exact half-integer ties. The tiny denominator addition prevents division by zero for a zero row.

| Row | Scale | Values before rounding, approximately | Stored integers | Reconstructed weights, approximately |
| --- | --- | --- | --- | --- |
| `[1, 0.4, -0.2]` | `1/127` | `[127, 50.8, -25.4]` | `[127, 51, -25]` | `[1, 0.401575, -0.196850]` |
| `[10, 4, -2]` | `10/127` | `[127, 50.8, -25.4]` | `[127, 51, -25]` | `[10, 4.015748, -1.968504]` |
| `[0, 0, 0]` | 0 | `[0, 0, 0]` | `[0, 0, 0]` | `[0, 0, 0]` |

Buffers are saved in `state_dict()` and follow module device moves, but optimizers do not discover them as trainable parameters.

```python
self.bias = nn.Parameter(bias.clone()) if bias is not None else None
```

An optional bias is cloned so its storage is separate from the supplied bias. Unlike quantized weight buffers, this bias is a trainable parameter.

```python
w = self.weight_int8.float() * self.scale
out = x @ w.T
if self.bias is not None:
    out = out + self.bias
return out
```

The integer tensor is converted back to float and multiplied by the per-row scales. `w.T` has shape `(I, O)`, so an input `(B, I)` produces `(B, O)`. A bias `(O,)` broadcasts across all leading input dimensions. This is a **floating-point** matrix multiply using reconstructed weights, not an integer-only computation.

For input `[1,1,1]`, the first output before bias is `1+51/127-25/127 = 153/127`, approximately `1.204724`. The original row would produce `1.2`; that small difference is quantization error.

## Run the row-scale example

Load the reference solution first, then run:

```python
# Example: Inspect row-wise integer weights and reconstructed outputs
weight = torch.tensor([[1.0, 0.4, -0.2],
                       [10.0, 4.0, -2.0],
                       [0.0, 0.0, 0.0]])
bias = torch.tensor([0.1, -0.2, 0.0])
layer = Int8Linear(weight, bias)
expected_ints = torch.tensor([[127, 51, -25], [127, 51, -25], [0, 0, 0]],
                             dtype=torch.int8)
assert torch.equal(layer.weight_int8, expected_ints)
torch.testing.assert_close(layer.scale, torch.tensor([[1/127], [10/127], [0.0]]))
x = torch.ones(1, 3)
expected = torch.tensor([[153/127 + 0.1, 1530/127 - 0.2, 0.0]])
torch.testing.assert_close(layer(x), expected)
assert layer(x).shape == (1, 3)
assert set(dict(layer.named_buffers())) == {'weight_int8', 'scale'}
assert set(dict(layer.named_parameters())) == {'bias'}
```

## Common mistakes and reference limits

A scale shaped `(O,)` broadcasts incorrectly across weight columns. Removing rounding truncates values instead of choosing nearest integers. The reference does not quantize activations or provide an INT8 kernel. Bias stays trainable. Construction does not explicitly detach the source weight, so its scale can retain an autograd link when the source requires gradients. Forward assumes compatible dtype and device.
