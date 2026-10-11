This function performs **gradient clipping by the global L2 norm**. It prevents gradients from becoming too large during training, which can help with **exploding gradients**.

It scales down the gradients when necessary—it does **not** change the model’s parameters directly.

## Line by line

### 1. Keep parameters that have gradients
```python
parameters = [p for p in parameters if p.grad is not None]
```
Each parameter `p` is usually a tensor of model weights or biases. After `loss.backward()`, its gradient is stored in `p.grad`.

Some parameters might not have gradients—for example, if they were not involved in computing the loss. This line excludes them.

### 2. Compute the combined size of all gradients
```python
total_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in parameters))
```
`p.grad.norm()` computes the L2 norm of one parameter’s gradient tensor.

The function squares each norm, adds them together, and takes the square root:

\[
\text{total\_norm}
= \sqrt{\sum_p \|\nabla_p L\|_2^2}
\]

This is equivalent to flattening **all gradient tensors into one long vector** and computing its L2 norm. It is a global norm, not a separate clipping threshold for each parameter.

### 3. Calculate a scaling factor
```python
clip_coef = max_norm / (total_norm + 1e-6)
```
The desired scale is approximately:

\[
\text{clip\_coef} = \frac{\text{maximum allowed norm}}{\text{current norm}}
\]

The small `1e-6` prevents division by zero.

- If the current norm exceeds `max_norm`, the factor is less than `1`, so the gradients should shrink.
- If the current norm is below `max_norm`, the factor is greater than `1`, but the function will **not** enlarge the gradients.

### 4. Scale the gradients in place, if needed
```python
if clip_coef < 1:
    for p in parameters:
        p.grad.mul_(clip_coef)
```
The trailing underscore in `mul_` means **in-place multiplication**: it modifies the existing gradient tensor.

Every gradient is multiplied by the **same factor**, preserving the direction of the combined gradient while reducing its magnitude to approximately `max_norm`.

### 5. Return the original norm
```python
return total_norm.item()
```
`.item()` converts the scalar PyTorch tensor into a Python number.

The returned value is the norm **before clipping**, which is useful for logging.

## Small example

Suppose all gradients together form the vector:

```text
[3, 4]
```

Its norm is:

```text
sqrt(3² + 4²) = 5
```

With `max_norm = 1`, the scaling factor is approximately `1 / 5 = 0.2`, producing:

```text
[0.6, 0.8]
```

The new norm is approximately `1`, but the function returns the original norm, `5.0`.

## Where it fits in training

```python
optimizer.zero_grad()
loss.backward()                            # Compute gradients
original_norm = clip_grad_norm(model.parameters(), max_norm=1.0)
optimizer.step()                           # Update weights using clipped gradients
```

**Practical note:** PyTorch already provides this operation:

```python
original_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)
```

The built-in version is generally preferable. One edge case in your implementation is that it fails if no parameters have gradients: `sum(...)` returns the Python integer `0`, and `torch.sqrt(0)` expects a tensor.