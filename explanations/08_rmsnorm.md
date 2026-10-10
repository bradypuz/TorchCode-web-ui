# Implement RMSNorm

RMSNorm divides a feature vector by a measure of its overall magnitude, then applies a learned scale per feature. There is no mean subtraction: the code preserves the vector's direction before applying those featurewise weights. The entire reference consists of one denominator calculation and one return statement.

## Give the dimensions names

Let `D` be features per vector. Input `x` has shape `(...,D)`, such as `(B,S,D)` for `B` examples with `S` tokens each. `weight` has shape `(D,)`, and output has the same shape as `x`. Each token is normalized independently across its final feature axis.

For `(2,3,4)`, the denominator has shape `(2,3,1)`. Every token has one scalar magnitude that broadcasts across its four features.

## 1. Read the magnitude calculation from inside out

```python
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
```

`x.pow(2)` squares every coordinate. This prevents positive and negative coordinates from canceling when measuring magnitude. For `[3,4]`, the squares are `[9,16]`.

`mean(dim=-1, keepdim=True)` averages across features, giving `(9+16)/2 = 12.5`. This is a mean square, not the sum of squares used by an ordinary Euclidean length. Keeping the axis gives shape `(1,)` for one vector or `(B,S,1)` for a token batch.

Adding `eps` before the square root protects the zero vector from a zero denominator. Finally, `sqrt` converts mean square to root mean square. Ignoring epsilon solely for this nonzero arithmetic example, `[3,4]` has RMS `sqrt(12.5) ≈ 3.5355`.

## 2. Divide each vector and apply feature weights

```python
return x / rms * weight
```

The scalar RMS for each vector broadcasts across its features. `[3,4]` therefore becomes approximately `[0.8485,1.1314]`. Both coordinates are divided by the same number, preserving their ratio. Multiplying by `weight=[2,0.5]` then changes the coordinates to approximately `[1.6971,0.5657]`.

The weights also broadcast across leading dimensions, but differently: there is one scale **per feature**, reused for all examples and tokens. The function receives those weights rather than creating a parameter itself.

| Input | Mean square | RMS with `eps=0` | Before feature weights |
|---|---:|---:|---|
| `[3,4]` | 12.5 | `sqrt(12.5)` | `[3/r,4/r]` |
| `[6,8]` | 50 | `2*sqrt(12.5)` | The same `[3/r,4/r]` |
| `[2,2]` | 4 | 2 | `[1,1]` |

The last row makes the absence of centering visible. A constant nonzero vector remains nonzero. Unit weights give a normalized mean square near one; they do not imply a zero mean.

```torchcode-visual
normalization
```

## Run the scale and gradient example

```python
# Example: Doubling a vector doubles its RMS too
x = torch.tensor([[3., 4.], [6., 8.]], requires_grad=True)
weight = torch.tensor([2., 0.5], requires_grad=True)
mean_square = x.pow(2).mean(dim=-1, keepdim=True)
assert torch.equal(mean_square, torch.tensor([[12.5], [50.]]))
out = rms_norm(x, weight, eps=0.)
r = 12.5 ** 0.5
expected = torch.tensor([[6. / r, 2. / r], [6. / r, 2. / r]])
assert torch.allclose(out, expected)
assert out.shape == (2, 2)
assert torch.equal(rms_norm(torch.tensor([[2., 2.]]), torch.ones(2), eps=0.), torch.ones(1, 2))
assert torch.equal(rms_norm(torch.zeros(1, 2), torch.ones(2)), torch.zeros(1, 2))
out.square().sum().backward()
assert x.grad is not None and weight.grad is not None
```

Using `eps=0` here makes the nonzero examples exactly scale-invariant. The zero-vector assertion instead uses the default positive epsilon. In real use, epsilon slightly changes that scale invariance near zero.

## Gradients and reference scope

Autograd differentiates both the direct input in the numerator and the input-dependent denominator. Coordinates within the same vector are coupled through the mean square; vectors remain independent. The reference has no bias, mean subtraction, module state, or explicit wider accumulation dtype. Squaring very large low-precision values can overflow. These limits do not change the basic sequence: square, average, add epsilon, square root, divide, and scale.
