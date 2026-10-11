This code implements a **fully connected layer** from scratch using PyTorch. It takes an input vector, multiplies it by trainable weights, and adds a trainable bias:

\[
y = xW^\top + b
\]

Here’s how each part works.

## 1. Creating the layer

```python
class SimpleLinear:
    def __init__(self, in_features: int, out_features: int):
```

The constructor takes:
- **`in_features`**: the number of values in each input.
- **`out_features`**: the number of values the layer produces for each input.

For example:

```python
layer = SimpleLinear(in_features=3, out_features=2)
```

This creates a layer that maps a vector with **3 features** to a vector with **2 features**.

The `: int` annotations indicate the expected argument types; they do not enforce those types at runtime.

## 2. Initializing the weights

```python
self.weight = torch.randn(out_features, in_features) * (1 / math.sqrt(in_features))
```

This creates a weight matrix with shape:

```text
(out_features, in_features)
```

For the example above, its shape is `(2, 3)`:

```text
              input 1   input 2   input 3
output 1       w₁₁       w₁₂       w₁₃
output 2       w₂₁       w₂₂       w₂₃
```

Each output has one weight for each input feature.

- `torch.randn(...)` samples random values from a normal distribution with mean `0` and standard deviation `1`.
- Multiplying by `1 / sqrt(in_features)` makes the weights smaller when there are more input features. This helps keep the initial output scale from growing simply because the input has more features.

```python
self.weight.requires_grad_(True)
```

This enables **gradient tracking** for the weights. During training, PyTorch can calculate how the loss changes with respect to each weight.

The trailing underscore in `requires_grad_` means it changes the existing tensor in place.

## 3. Initializing the bias

```python
self.bias = torch.zeros(out_features, requires_grad=True)
```

This creates one bias value for each output, initially set to zero.

For a layer with two outputs:

```text
bias = [0, 0]
```

The bias is trainable too, because `requires_grad=True`.

## 4. Computing the output

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x @ self.weight.T + self.bias
```

This is the layer’s **forward pass**.

- `@` performs matrix multiplication.
- `self.weight.T` transposes the weight matrix.
- `+ self.bias` adds the bias to each output.

For a batch of inputs, the shapes are:

```text
x:             (batch_size, in_features)
weight:        (out_features, in_features)
weight.T:      (in_features, out_features)

x @ weight.T:  (batch_size, out_features)
bias:          (out_features,)
```

PyTorch automatically **broadcasts** the bias across all examples in the batch.

For a single input with three features, each output is computed as:

```text
output₁ = x₁w₁₁ + x₂w₁₂ + x₃w₁₃ + b₁
output₂ = x₁w₂₁ + x₂w₂₂ + x₃w₂₃ + b₂
```

## Example

```python
layer = SimpleLinear(3, 2)

# Four examples, each containing three features
x = torch.randn(4, 3)

y = layer.forward(x)

print(y.shape)
# torch.Size([4, 2])
```

Each of the four input examples produces two output values.

## Important details

- **There is no activation function.** This only performs a weighted sum plus bias. Strictly speaking, it is an *affine* transformation, although neural-network terminology calls it a linear layer.
- **Gradients do not update parameters automatically.** Calling `loss.backward()` computes gradients in `layer.weight.grad` and `layer.bias.grad`; an optimizer or manual update must then change the parameters.
- **This is an ordinary Python class, not a `torch.nn.Module`.** You call `layer.forward(x)`, not `layer(x)`, and it lacks conveniences such as `.parameters()` and `.to(device)`.

The standard PyTorch counterpart is:

```python
layer = torch.nn.Linear(3, 2)
y = layer(x)
```

It performs the same kind of computation, though its default initialization differs.