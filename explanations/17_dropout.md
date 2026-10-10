# Dropout

## What the layer does

During training, independently set some input elements to zero. This makes a model practice working with different subsets of its intermediate features. During evaluation, pass every element through unchanged.

`x` can be a floating-point tensor of any shape. The output has that same shape. The parameter `p` is the probability of dropping an element; `1 - p` is its probability of surviving. For this implementation use `0 <= p < 1`. Unlike a learned linear layer, dropout has no weights or biases to optimize.

## Store the probability and respect the module's mode

```python
super().__init__()
self.p = p
```

The parent constructor gives the object its PyTorch module behavior, including the `training` flag. A newly created module starts in training mode. Calling `.eval()` switches that flag off, and `.train()` switches it on again. The constructor stores the requested probability but does not validate it.

The first part of `forward` handles the cases that need no randomness:

```python
if not self.training or self.p == 0:
    return x
```

Evaluation and zero dropout both return the original tensor itself. A model can therefore have a dropout layer present while evaluation remains deterministic. `torch.no_grad()` alone does **not** activate this branch: it disables gradient recording, not the module's training flag.

## Make one random decision per element

```python
mask = (torch.rand_like(x) > self.p).float()
```

`torch.rand_like(x)` draws numbers between zero and one with `x`'s shape and device. Comparing each number with `p` produces `True` for survivors and `False` for dropped values. `.float()` turns these into `1.0` and `0.0`, allowing multiplication. All dimensions participate in the random draw: there is a separate decision for each batch item, token, and feature.

For `p = 0.5`, a draw of `0.77` survives, while `0.09` is dropped. Drawing again usually gives a different mask. A manual random seed makes the sequence of draws reproducible, which is useful when explaining or testing the calculation; reusing the same mask every training step would change the behavior.

## Why surviving values get larger

```python
return x * mask / (1 - self.p)
```

Multiplication removes the dropped elements. Division then enlarges survivors by `1 / (1 - p)`. At `p = 0.5`, survivors double. For an original value `6`, there is roughly a half chance of returning `0` and a half chance of returning `12`; the average is still `6`. More generally, the expected value is `x * (1 - p) / (1 - p) = x`.

This preserved average is why evaluation can simply return `x`. The scale is applied during training, a convention called **inverted dropout**. A particular small tensor's sum need not be preserved: the guarantee concerns the average over random masks.

## Inspect the mask and scale

With seed zero, the first six CPU draws are approximately as follows:

| Input | Draw | Keep when draw > 0.5? | Training output |
|---|---|---|---|
| `1` | `0.4963` | No | `0` |
| `2` | `0.7682` | Yes | `4` |
| `3` | `0.0885` | No | `0` |
| `4` | `0.1320` | No | `0` |
| `5` | `0.3074` | No | `0` |
| `6` | `0.6341` | Yes | `12` |

```torchcode-visual
dropout
```

Use the visual to change `p` and compare the keep mask, training scale, and evaluation output. The executable example below checks an actual draw from the reference code.

```python
# Example: Drop four entries and double the two survivors
layer = MyDropout(p=0.5)
x = torch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
torch.manual_seed(0)
y = layer(x)
assert y.shape == x.shape
assert torch.equal(y, torch.tensor([[0., 4., 0.], [0., 0., 12.]]))
y.sum().backward()
assert torch.equal(x.grad, torch.tensor([[0., 2., 0.], [0., 0., 2.]]))
layer.eval()
assert layer(x) is x
layer.train()
layer.p = 0
assert layer(x) is x
```

The gradients use the same mask and scaling: a dropped element has zero gradient, and a surviving element has gradient two for this sum loss. Evaluation and `p = 0` both preserve the input exactly.

## Common mistakes and reference scope

Do not scale the evaluation output, or confuse the drop probability with the keep probability. Use `.eval()` when evaluating the containing model. This reference does not handle `p = 1`: it divides by zero. Its `.float()` creates a float32 mask, so lower-precision inputs can be promoted instead of preserving the exact input dtype.
