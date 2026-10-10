# Cosine Learning Rate Schedule with Warmup

This function answers a small question: what learning rate should training use at a particular step? It starts with a straight ramp to the maximum, follows a smooth cosine curve down to the minimum, and stays at that minimum after training ends.

## Inputs and output

`step`, `total_steps`, and `warmup_steps` are scalar counts. `max_lr` and `min_lr` are scalar learning rates. The output is one number, not a tensor. There is no stored counter or optimizer inside the function: the training loop supplies `step` and assigns the returned value to the optimizer.

## Reading the actual code

```python
if step < warmup_steps:
    return max_lr * step / warmup_steps
```

The first branch is linear warmup. For `warmup_steps=2` and `max_lr=0.1`, step zero returns `0`, and step one returns `0.05`. The strict `<` comparison means step two leaves this branch. A zero-length warmup is safe for nonnegative steps because this branch is never entered.

```python
if step >= total_steps:
    return min_lr
```

This branch stops the schedule at the endpoint. Without it, extending a cosine past its intended interval would eventually make the learning rate rise again. With `total_steps=6`, step six and every later step return the minimum.

```python
progress = (step - warmup_steps) / (total_steps - warmup_steps)
```

Only the remaining decay interval belongs in `progress`. In our six-step run with two warmup steps, decay lasts four steps. Step two has progress `0/4 = 0`; step four has progress `2/4 = 0.5`. Using `step/total_steps` would start decay partway through its curve and miss the intended peak.

```python
return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
```

`cos(pi*progress)` goes from `1` to `-1`. Adding one and halving maps it to a factor from `1` down to `0`. Multiplying by the learning-rate range and adding `min_lr` sets the two endpoints. At progress `0.5`, cosine is zero, so the factor is `0.5` and the learning rate is halfway between the bounds.

| Step | Branch | Decay progress | Learning rate with bounds 0.02 and 0.10 |
| ---: | --- | ---: | ---: |
| 0 | Warmup | — | 0.00 |
| 1 | Warmup | — | 0.05 |
| 2 | Cosine | 0.00 | 0.10 |
| 3 | Cosine | 0.25 | approximately 0.088284 |
| 4 | Cosine | 0.50 | 0.06 |
| 5 | Cosine | 0.75 | approximately 0.031716 |
| 6 | Finished | — | 0.02 |

Warmup begins at zero even when the eventual minimum is `0.02`; `min_lr` belongs to decay and the finished branch.

## Run the schedule

Load the reference solution first, then run:

```python
# Example: Check warmup, cosine midpoint, and the finished schedule
rates = [cosine_lr_schedule(s, 6, 2, 0.1, 0.02) for s in range(8)]
expected = [0.0, 0.05, 0.1, 0.0882842712474619,
            0.06, 0.0317157287525381, 0.02, 0.02]
assert all(math.isclose(a, b, abs_tol=1e-12)
           for a, b in zip(rates, expected))
assert cosine_lr_schedule(0, 4, 0, 0.1) == 0.1
```

The second assertion confirms that disabling warmup starts directly at the maximum.

## Common mistakes and reference limits

Use a valid schedule with `0 <= warmup_steps < total_steps`. The code does not validate counts, negative steps, or rate ordering. The warmup branch is checked first, so a warmup longer than the whole run can override the finished branch. Calling the function alone does not change any optimizer parameter group.
