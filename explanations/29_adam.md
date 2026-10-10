# Adam Optimizer

Adam updates a parameter using two memories of its past gradients. The first remembers the gradient's direction; the second remembers its size squared. Dividing the first by the square root of the second makes the update sensitive to a parameter's own gradient history.

## Inputs, outputs, and stored state

`MyAdam(params, lr, betas, eps)` takes an iterable of parameter tensors. Each parameter can have any shape. Its gradient `p.grad` and the corresponding state tensors `m[i]` and `v[i]` have that same shape, dtype, and device. `step()` changes parameters in place and returns `None`. `zero_grad()` sets existing gradients to zero.

## Reading the actual code

```python
self.params = list(params)
self.beta1, self.beta2 = betas
self.t = 0
self.m = [torch.zeros_like(p) for p in self.params]
self.v = [torch.zeros_like(p) for p in self.params]
```

Materializing the iterable matters because optimizers revisit parameters every step; a generator can otherwise be exhausted. The two beta values control how much old memory survives. With the defaults, `m` keeps 90% of its previous value and `v` keeps 99.9%. `zeros_like` creates a separate state tensor for every parameter, including matrices and scalars. `lr` and `eps` are also stored on the object. Reuse this object across training steps so these memories survive.

```python
self.t += 1
with torch.no_grad():
    for i, p in enumerate(self.params):
        if p.grad is None:
            continue
```

The counter advances once per call, not once per tensor. `enumerate` connects parameter `p` to its matching state entries. A parameter with no gradient is skipped. `no_grad()` prevents the update itself from becoming part of the next backward graph: the optimizer consumes gradients that training has already computed.

```python
self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad ** 2
```

These operations are elementwise. Suppose one parameter starts at `1`, its first gradient is `2`, and both memories start at zero. The default betas produce `m = 0.9*0 + 0.1*2 = 0.2` and `v = 0.999*0 + 0.001*4 = 0.004`. Squaring prevents opposite gradient signs from cancelling in the size estimate.

```python
m_hat = self.m[i] / (1 - self.beta1 ** self.t)
v_hat = self.v[i] / (1 - self.beta2 ** self.t)
p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
```

The zero initialization makes early moving averages too small. At step one, bias correction divides `0.2` by `0.1` and `0.004` by `0.001`, giving `m_hat = 2` and `v_hat = 4`. With `lr=0.1`, the update is approximately `0.1*2/sqrt(4) = 0.1`, so the parameter becomes `0.9`. Epsilon sits **outside** the square root and protects the final division when the estimated gradient size is zero.

| First-step quantity | Coordinate 0 | Coordinate 1 |
| --- | ---: | ---: |
| Initial parameter | 1 | -1 |
| Gradient | 2 | -4 |
| Stored `m` | 0.2 | -0.4 |
| Stored `v` | 0.004 | 0.016 |
| Corrected `m_hat` | 2 | -4 |
| Corrected `v_hat` | 4 | 16 |
| Parameter after `lr=0.1` | approximately 0.9 | approximately -0.9 |

```python
for p in self.params:
    if p.grad is not None:
        p.grad.zero_()
```

This is the entire `zero_grad()` implementation. It preserves allocated gradient tensors. Backward adds into existing gradients, so clearing them before the next ordinary training step prevents accidentally combining steps. Clearing gradients does not clear Adam's memories.

## Run a first-step example

Load the reference solution first, then run:

```python
# Example: Inspect Adam's first update and persistent memories
p = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
optimizer = MyAdam([p], lr=0.1)
p.grad = torch.tensor([2.0, -4.0])
optimizer.step()
assert optimizer.t == 1
torch.testing.assert_close(optimizer.m[0], torch.tensor([0.2, -0.4]))
torch.testing.assert_close(optimizer.v[0], torch.tensor([0.004, 0.016]))
torch.testing.assert_close(p, torch.tensor([0.9, -0.9]))
optimizer.zero_grad()
assert torch.equal(p.grad, torch.zeros(2))
assert torch.count_nonzero(optimizer.m[0]).item() == 2
```

Both coordinates move approximately `0.1`, despite different gradient magnitudes: first-step normalization divides each gradient by its own absolute magnitude. Later updates depend on accumulated history.

## Common mistakes and reference limits

Calling `step()` does not clear gradients. Recreating the optimizer each iteration loses its state. The reference has one global counter; if a tensor starts receiving gradients later, its bias correction still uses the global step, unlike PyTorch Adam's per-parameter counters. It omits parameter groups, weight decay, sparse gradients, and hyperparameter validation.
