This module implements a **weight-only, per-output-channel int8-quantized linear layer**. It stores each row of the weight matrix as signed 8-bit integers plus a floating-point scale, then reconstructs an approximate floating-point weight during every forward pass.

It behaves approximately like:

```python
nn.Linear(in_features, out_features, bias=True_or_false)
```

but the weight is stored in compressed int8 form.

---

## Expected shapes

Assume:

```python
weight.shape == (out_features, in_features)
bias.shape   == (out_features,)       # if provided
x.shape      == (..., in_features)
```

The output shape is:

```python
(..., out_features)
```

---

## Initialization

```python
class Int8Linear(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
```

The constructor receives an existing floating-point weight tensor and an optional bias.

For a regular `nn.Linear`, those could come from:

```python
linear = nn.Linear(128, 64)
quantized = Int8Linear(linear.weight, linear.bias)
```

Here, `linear.weight` has shape:

```text
(64, 128)
```

Each of its 64 rows corresponds to one output feature/neuron.

---

## 1. Compute one scale per output feature

```python
scale = weight.abs().amax(dim=1, keepdim=True) / 127
```

This performs **symmetric per-row quantization**.

Breaking it down:

```python
weight.abs()
```

takes the absolute value of every weight.

```python
.amax(dim=1, keepdim=True)
```

finds the largest absolute value in each row.

If:

```python
weight.shape == (out_features, in_features)
```

then:

```python
scale.shape == (out_features, 1)
```

Because `keepdim=True`, the second dimension remains present. This makes broadcasting work when dividing or multiplying by the scale.

The division by `127` comes from the chosen quantized range:

```text
-127 through +127
```

The largest-magnitude floating-point weight in each row is mapped approximately to `-127` or `127`.

Mathematically, for output row \(i\):

\[
s_i = \frac{\max_j |W_{ij}|}{127}
\]

Each row gets its own scale, which generally produces lower quantization error than using a single scale for the entire weight matrix.

---

## 2. Prevent zero scales

```python
scale = scale.clamp(min=1e-10)
```

If an entire weight row contains zeros, its maximum absolute value is zero, so the scale would otherwise be zero.

Later, the code divides by the scale:

```python
weight / scale
```

Dividing by zero would produce invalid values. Clamping ensures every scale is at least:

```text
1e-10
```

For an all-zero row, the quantized values still become zero.

---

## 3. Quantize and store the weight

```python
self.register_buffer(
    'weight_int8',
    (weight / scale).round().clamp(-127, 127).to(torch.int8)
)
```

Conceptually, the quantization formula is:

\[
Q_{ij} =
\operatorname{clamp}
\left(
\operatorname{round}\left(\frac{W_{ij}}{s_i}\right),
-127,
127
\right)
\]

Step by step:

### Divide by the row scale

```python
weight / scale
```

The scale has shape `(out_features, 1)`, so PyTorch broadcasts it across every element in the corresponding row.

This converts each floating-point weight into its value in quantized units.

For example, if one row has:

```python
scale = 0.01
weight = [0.25, -0.70, 1.00]
```

then:

```python
weight / scale = [25, -70, 100]
```

### Round to the nearest integer

```python
.round()
```

Int8 values cannot store fractional values, so the results are rounded.

For example:

```text
25.3  -> 25
-69.7 -> -70
```

### Clamp the range

```python
.clamp(-127, 127)
```

This ensures the results fit the selected symmetric int8 range.

Although `torch.int8` can represent values from `-128` through `127`, this code deliberately uses `-127` through `127`. Using equal positive and negative magnitudes simplifies symmetric quantization.

### Convert to int8

```python
.to(torch.int8)
```

This stores each quantized weight using one byte instead of the typical four bytes used by `float32`.

Ignoring scale overhead, this reduces the weight storage by roughly 4×:

```text
float32: 4 bytes per weight
int8:    1 byte per weight
```

---

## Why `register_buffer` is used

```python
self.register_buffer('weight_int8', ...)
self.register_buffer('scale', scale)
```

A registered buffer:

- belongs to the module;
- is included in `state_dict()`;
- moves with the module when calling `.to(device)`, `.cuda()`, or `.cpu()`;
- is not normally optimized by an optimizer;
- is not treated as a trainable parameter.

For example:

```python
layer.cuda()
```

moves both `weight_int8` and `scale` to the GPU.

But:

```python
optimizer = torch.optim.Adam(layer.parameters())
```

does not include those buffers as ordinary trainable weights.

The intended meaning is that the quantized weight is fixed.

---

## 4. Store an optional trainable bias

```python
self.bias = nn.Parameter(bias.clone()) if bias is not None else None
```

If a bias is provided:

1. it is cloned;
2. it is wrapped in `nn.Parameter`;
3. optimizers will treat it as trainable.

Thus, this layer has:

- a frozen int8 weight;
- a frozen scale;
- an optionally trainable floating-point bias.

If no bias is passed, `self.bias` is `None`.

---

# Forward pass

## 1. Dequantize the weight

```python
w = self.weight_int8.float() * self.scale
```

The stored int8 weights are first converted to floating point:

```python
self.weight_int8.float()
```

Then each row is multiplied by its scale:

\[
\hat{W}_{ij} = Q_{ij}s_i
\]

This produces an approximate reconstruction of the original weight.

For example:

```text
original weight:       0.253
scale:                 0.010
quantized value:       round(0.253 / 0.010) = 25
reconstructed weight:  25 × 0.010 = 0.250
```

The difference:

```text
0.253 - 0.250 = 0.003
```

is quantization error.

Because rounding was used, the maximum error for a value that was not clipped is normally about half of one quantization step:

\[
|\text{error}| \lesssim \frac{s_i}{2}
\]

---

## 2. Apply the linear transformation

```python
x = x @ w.T
```

The stored weight follows the `nn.Linear` convention:

```python
w.shape == (out_features, in_features)
```

But matrix multiplication expects the input dimension to match the first matrix dimension being multiplied. Therefore, the weight is transposed:

```python
w.T.shape == (in_features, out_features)
```

So:

```text
x:   (..., in_features)
w.T: (in_features, out_features)
--------------------------------
out: (..., out_features)
```

This is equivalent to:

```python
torch.nn.functional.linear(x, w, bias=None)
```

or mathematically:

\[
Y = X\hat{W}^{T}
\]

---

## 3. Add the bias

```python
if self.bias is not None:
    x += self.bias
```

The bias has shape:

```python
(out_features,)
```

PyTorch broadcasts it across all leading dimensions of the output.

For a batch:

```text
x output shape: (batch_size, out_features)
bias shape:     (out_features,)
```

the same bias is added to every example.

The final computation is:

\[
Y = X\hat{W}^{T} + b
\]

---

# Small numerical example

Suppose one output row is:

```python
weight = torch.tensor([[0.2, -0.5, 1.0]])
```

The maximum absolute value is:

```text
max(abs(weight)) = 1.0
```

The scale is:

```text
scale = 1.0 / 127 ≈ 0.007874
```

Quantization gives approximately:

```text
0.2  / 0.007874 ≈  25.4 ->  25
-0.5 / 0.007874 ≈ -63.5 -> -64
1.0  / 0.007874 = 127.0 -> 127
```

So the stored int8 row is:

```python
[25, -64, 127]
```

Dequantization reconstructs:

```text
25  × 0.007874 ≈  0.19685
-64 × 0.007874 ≈ -0.50394
127 × 0.007874 =  1.00000
```

This is close to the original:

```text
original:      [ 0.20000, -0.50000, 1.00000]
reconstructed: [ 0.19685, -0.50394, 1.00000]
```

---

# Important limitation: this is not true int8 matrix multiplication

Despite the name `Int8Linear`, the forward pass does this:

```python
w = self.weight_int8.float() * self.scale
x = x @ w.T
```

The int8 weight is converted back to floating point before matrix multiplication. Therefore:

- storage is quantized;
- the multiplication is floating-point;
- it does not directly use an int8 GEMM kernel;
- it may not produce the speedup expected from real int8 inference;
- it allocates/materializes a floating-point weight during every forward pass.

So this is best described as:

> int8 weight storage with on-the-fly floating-point dequantization

rather than a fully quantized linear operator.

True int8 inference usually quantizes the activations too and performs integer matrix multiplication, followed by appropriately scaled accumulation. Optimized implementations use backends such as PyTorch quantization operators, FBGEMM, QNNPACK, oneDNN, cuBLASLt, or specialized kernels.

---

# Training behavior

The original full-precision weight is not stored as an `nn.Parameter`. Instead, the module stores:

```python
weight_int8  # buffer
scale        # buffer
bias         # parameter, if present
```

Consequently:

- the quantized weight is not normally updated by an optimizer;
- rounding and int8 conversion are not differentiable;
- only the bias is intentionally trainable;
- gradients can still flow through the matrix multiplication into `x`.

For example, if `x.requires_grad=True`, gradients can be computed for `x`, but the int8 weight itself will not be trained like an ordinary `nn.Linear.weight`.

This makes the class primarily suited to inference or to a model in which only the biases or other surrounding parameters are being fine-tuned.

---

# One subtle implementation issue

If the supplied `weight` is an `nn.Parameter` with `requires_grad=True`, then `scale` is derived from it before being registered as a buffer:

```python
scale = weight.abs().amax(...) / 127
self.register_buffer('scale', scale)
```

Registering a tensor as a buffer does not necessarily detach its existing autograd history. For a cleaner inference-only conversion, the quantization should generally happen using a detached weight or inside `torch.no_grad()`:

```python
with torch.no_grad():
    weight = weight.detach()
    scale = weight.abs().amax(dim=1, keepdim=True).div(127)
    scale = scale.clamp_min(1e-10)
    weight_int8 = (
        weight.div(scale)
        .round()
        .clamp(-127, 127)
        .to(torch.int8)
    )

self.register_buffer("weight_int8", weight_int8)
self.register_buffer("scale", scale)
```

That makes the intended frozen nature of both buffers explicit and avoids keeping an unnecessary connection to the source parameter’s computation graph.

---

# Dtype consideration

This line explicitly calls:

```python
self.weight_int8.float()
```

which produces `float32`. In ordinary PyTorch execution, the input and weight used by matrix multiplication generally need compatible dtypes.

Thus, an input such as:

```python
x.dtype == torch.float16
```

may create a dtype mismatch with the reconstructed float32 weight, depending on the device and autocast context.

A dtype-aware version could reconstruct the weight in the input dtype:

```python
w = self.weight_int8.to(dtype=x.dtype) * self.scale.to(dtype=x.dtype)
```

However, using very small scales in `float16` can have numerical consequences, so production code should choose the compute dtype deliberately.

---

# Memory implications

The persistent weight representation uses:

- one byte per element for `weight_int8`;
- one floating-point scale per output row;
- an optional floating-point bias.

For a weight matrix with shape:

```text
(out_features, in_features)
```

the approximate persistent storage is:

```text
out_features × in_features bytes
+ out_features × scale_element_size
```

For large matrices, the per-row scale overhead is small.

However, during every forward pass, this line recreates a full floating-point matrix:

```python
w = self.weight_int8.float() * self.scale
```

Therefore, peak runtime memory still includes a floating-point copy of the weight. The module reduces stored model size, but does not retain the full memory benefit throughout computation.

---

# Summary

The module follows this process:

### At construction

\[
s_i = \max\left(\frac{\max_j |W_{ij}|}{127}, 10^{-10}\right)
\]

\[
Q_{ij} =
\operatorname{int8}
\left(
\operatorname{clamp}
\left(
\operatorname{round}(W_{ij}/s_i),
-127,
127
\right)
\right)
\]

It stores:

```text
Q: int8 quantized weight
s: floating-point scale for every output row
b: optional floating-point trainable bias
```

### At inference

\[
\hat{W}_{ij} = Q_{ij}s_i
\]

\[
Y = X\hat{W}^{T} + b
\]

In short, it is a simple and understandable example of **symmetric, per-output-channel weight quantization**, but it provides mainly **compressed weight storage**, not true end-to-end int8 computation.