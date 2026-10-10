# Simple Linear Layer

The class learns a weighted combination of input features for every output feature. Its constructor creates those weights and biases; `forward` uses them. The key to understanding the code is the orientation of the stored weight matrix.

## Assign a meaning to each dimension

Let `I = in_features`, `O = out_features`, and `B` be batch size. A typical input is `(B, I)`, the weight is `(O, I)`, the bias is `(O,)`, and the output is `(B, O)`. Each row of the weight contains all input coefficients for **one output neuron**.

```text
input                     stored weight       transposed weight
(B, I)                    (O, I)              (I, O)
   └────────────────────────────── @ ─────────────┘
                                      │
                                  (B, O) + bias (O,)
                                      │
                                  output (B, O)
```

The batch axis survives the multiplication. The input-feature axis is summed away, and the output-feature axis replaces it.

## 1. Initialize the weight at the requested scale

```python
self.weight = torch.randn(out_features, in_features) * (1 / math.sqrt(in_features))
self.weight.requires_grad_(True)
```

`randn` draws independent normal values with standard deviation one. Scaling by `1/sqrt(I)` makes individual weights smaller when more input terms will be added together. This is the exercise's specified initialization; it is not a call to the default `nn.Linear` initializer.

The second statement enables gradient tracking after initialization. Because the source random tensor did not require gradients, the resulting weight is a leaf tensor. Its `.grad` can therefore store the gradient of a loss during `backward()`.

## 2. Create an output-sized bias

```python
self.bias = torch.zeros(out_features, requires_grad=True)
```

There is one bias per output neuron, not one per input feature or batch example. Starting at zero makes the initial transformation's shift zero. Setting `requires_grad=True` lets training learn those shifts independently from the weight coefficients.

## 3. Multiply with the transpose, then broadcast the bias

```python
return x @ self.weight.T + self.bias
```

For one sample `[1, 2]`, suppose the weight rows are `[1, 2]` and `[-1, 1]`, with bias `[0.5, -0.5]`. The first output is `1*1 + 2*2 + 0.5 = 5.5`. The second is `1*(-1) + 2*1 - 0.5 = 0.5`.

The transpose is necessary because matrix multiplication matches the final axis of `x` with the first axis of its right operand. Adding `(O,)` to `(B, O)` broadcasts the same bias across every batch row. For token inputs `(B, S, I)`, the same expression also works: every token gets the same learned transformation and an output `(B, S, O)`.

## Run exact arithmetic and inspect the gradients

```python
# Example: Two samples use the same weight rows and bias
layer = SimpleLinear(2, 2)
with torch.no_grad():
    layer.weight.copy_(torch.tensor([[1., 2.], [-1., 1.]]))
    layer.bias.copy_(torch.tensor([0.5, -0.5]))
x = torch.tensor([[1., 2.], [-1., 3.]], requires_grad=True)
y = layer.forward(x)
assert torch.allclose(y, torch.tensor([[5.5, 0.5], [5.5, 3.5]]))
y.sum().backward()
assert torch.equal(layer.weight.grad, torch.tensor([[0., 5.], [0., 5.]]))
assert torch.equal(layer.bias.grad, torch.tensor([2., 2.]))
assert torch.equal(x.grad, torch.tensor([[0., 3.], [0., 3.]]))
```

For the summed-output loss, each weight row receives the sum of the two inputs: `[1 + (-1), 2 + 3] = [0, 5]`. Each bias appears twice, so each bias gradient is two. The input gradient sums the two weight rows, giving `[0, 3]` for each sample.

## Reference scope

This is a plain Python class, not an `nn.Module`. Call `.forward(...)` explicitly and manage its tensors yourself; the class has no automatic `.parameters()`, `.to()`, or `state_dict()` methods. It assumes a positive input-feature count and uses the current default device and dtype during construction. The core affine operation and its gradients are fully represented by the three statements above.
