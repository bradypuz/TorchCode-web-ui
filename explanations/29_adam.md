This code implements a simplified **Adam optimizer** in PyTorch. An optimizer updates a model’s parameters using the gradients computed by `loss.backward()`, with the aim of reducing the loss.

Adam combines two ideas:

- **Momentum:** smooth the gradients so updates are less sensitive to noisy individual steps.
- **Adaptive scaling:** normalize updates using a running estimate of each parameter element’s gradient magnitude.

Here’s how the code works.

## 1. Initialization: `__init__`

```python
def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
```

The arguments are:

| Argument | Meaning |
|---|---|
| `params` | The tensors to optimize, usually supplied by `model.parameters()`. |
| `lr` | Learning rate: controls the overall size of updates. Default: `0.001`. |
| `betas` | Decay rates for the two running averages. Values closer to `1` retain more history. |
| `eps` | A small number added to the denominator to avoid division by zero and improve numerical stability. |

### Store the parameters and settings

```python
self.params = list(params)
self.lr = lr
self.beta1, self.beta2 = betas
self.eps = eps
self.t = 0
```

- `list(params)` turns the parameter iterable into a reusable list. It keeps references to the original tensors—it does **not** copy their values.
- `self.t` counts calls to `step()` and is used for bias correction.

### Create the optimizer’s memory

```python
self.m = [torch.zeros_like(p) for p in self.params]
self.v = [torch.zeros_like(p) for p in self.params]
```

For every parameter tensor, Adam stores two tensors:

- **`m`**: a running average of gradients, called the *first moment*.
- **`v`**: a running average of squared gradients, called the *second moment*. It is not directly the variance.

`torch.zeros_like(p)` creates zeros with the same shape, device, and data type as `p`. Each element of each parameter has its own running averages.

## 2. Updating the parameters: `step()`

This method should be called **after gradients have been computed**.

### A. Advance the step counter

```python
self.t += 1
```

The first call uses `t = 1`, the second uses `t = 2`, and so on.

### B. Disable gradient tracking for the updates

```python
with torch.no_grad():
```

Updating parameters is part of optimization, not part of the forward computation whose derivatives we want to calculate.

`torch.no_grad()` prevents PyTorch from building a computation graph for these updates. It does **not** erase existing gradients.

### C. Process each parameter

```python
for i, p in enumerate(self.params):
    if p.grad is None:
        continue
```

`p.grad` contains the gradient of the loss with respect to `p`, normally populated by:

```python
loss.backward()
```

If a parameter has no gradient, the optimizer skips it.

### D. Update the running average of gradients

```python
self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
```

Mathematically, writing the current gradient as \(g_t\):

\[
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
\]

With the default `beta1 = 0.9`, this means:

```text
new m = 90% of old m + 10% of current gradient
```

This smooths the update direction over time—the momentum part of Adam.

### E. Update the running average of squared gradients

```python
self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad ** 2
```

Mathematically:

\[
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
\]

With `beta2 = 0.999`:

```text
new v = 99.9% of old v + 0.1% of current squared gradient
```

Squaring makes the values nonnegative and tracks gradient magnitude regardless of sign.

All these operations are **elementwise**, not matrix multiplications.

### F. Correct the initial bias

```python
m_hat = self.m[i] / (1 - self.beta1 ** self.t)
v_hat = self.v[i] / (1 - self.beta2 ** self.t)
```

Because `m` and `v` start at zero, their early values are biased toward zero. These divisions compensate for that initialization:

\[
\hat m_t = \frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t = \frac{v_t}{1-\beta_2^t}
\]

For example, on the first step:

\[
m_1 = (1-\beta_1)g_1
\]

so bias correction gives:

\[
\hat m_1 = g_1
\]

Likewise, \(\hat v_1 = g_1^2\).

### G. Apply the Adam update

```python
p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
```

This implements:

\[
\boxed{
p_t = p_{t-1}
- \text{lr}\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
}
\]

The numerator supplies the smoothed gradient; the denominator normalizes it using recent gradient magnitudes.

Unlike ordinary gradient descent,

```python
p -= lr * p.grad
```

Adam uses gradient history to determine each element’s update.

The `-=` changes the parameter **in place**, so the model sees the updated values.

## 3. Clearing gradients: `zero_grad()`

```python
def zero_grad(self):
    for p in self.params:
        if p.grad is not None:
            p.grad.zero_()
```

PyTorch **accumulates gradients** across calls to `backward()`. It does not automatically replace them.

This method clears existing gradient tensors so the next training iteration starts fresh. The underscore in `zero_()` means it modifies the tensor in place.

Importantly, it clears only `p.grad`. It preserves Adam’s running averages, `m` and `v`, because those need to carry information across training steps.

## 4. Typical usage

```python
optimizer = MyAdam(model.parameters(), lr=1e-3)

for inputs, targets in dataloader:
    optimizer.zero_grad()           # Clear previous gradients

    predictions = model(inputs)     # Forward pass
    loss = loss_fn(predictions, targets)

    loss.backward()                 # Compute parameter gradients
    optimizer.step()                # Update parameters
```

**`backward()` computes the gradients; `step()` uses them.**

## 5. Important limitations

This is a useful educational implementation, but it is not a complete replacement for `torch.optim.Adam`:

- **It uses one global step counter.** PyTorch Adam tracks steps separately for each parameter. If some parameters intermittently have `grad is None`, the bias corrections can differ.
- It does not implement features such as parameter groups, weight decay, AMSGrad, or optimizer checkpoint methods.
- It zeros gradients rather than setting them to `None`. Those are not identical: a parameter with a zero gradient can still move because of stored momentum, whereas this code skips a parameter whose gradient is `None`.

**In short:** the optimizer remembers both the direction and magnitude of past gradients, corrects its initially zero-valued estimates, and uses them to make adaptive parameter updates.