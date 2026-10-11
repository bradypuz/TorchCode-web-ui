This function calculates the **learning rate at a given training step** using two phases: **linear warmup**, followed by **cosine decay**.

The learning rate controls how large an update the optimizer makes to a model’s parameters. This schedule starts with small updates, increases them, then gradually decreases them toward the end of training.

## Parameters

- **`step`**: The current training step.
- **`total_steps`**: The number of steps planned for training.
- **`warmup_steps`**: How many initial steps to spend increasing the learning rate.
- **`max_lr`**: The peak learning rate.
- **`min_lr`**: The final learning rate; defaults to `0.0`.

## 1. Linear warmup

```python
if step < warmup_steps:
    return max_lr * step / warmup_steps
```

During warmup, the learning rate increases proportionally to the current step:

\[
\text{lr} = \text{max\_lr} \times \frac{\text{step}}{\text{warmup\_steps}}
\]

For example, if `warmup_steps = 100` and `max_lr = 0.001`:

- At step `0`, the learning rate is `0`.
- At step `50`, it is `0.0005`.
- At step `100`, it reaches `0.001` through the cosine branch below.

**Why warm up?** Small initial updates can help avoid unstable training when the model and optimizer are just starting.

## 2. Stop decreasing after training ends

```python
if step >= total_steps:
    return min_lr
```

At or beyond the final training step, the function returns `min_lr`.

This also prevents the cosine curve from cycling upward again if the function is called after training is supposed to end.

## 3. Cosine decay

```python
progress = (step - warmup_steps) / (total_steps - warmup_steps)
```

This measures progress through the **post-warmup portion** of training:

- `progress = 0` at the end of warmup.
- `progress = 0.5` halfway through the decay phase.
- `progress` approaches `1` as training approaches its final step.

Then:

```python
return min_lr + 0.5 * (max_lr - min_lr) * (
    1.0 + math.cos(math.pi * progress)
)
```

The cosine term smoothly lowers the learning rate:

| Decay progress | `cos(π × progress)` | Learning rate |
|---|---:|---|
| `0` | `1` | `max_lr` |
| `0.5` | `0` | Halfway between `max_lr` and `min_lr` |
| `1` | `-1` | `min_lr` |

Unlike a straight-line decrease, cosine decay changes slowly near the beginning and end of the decay phase, and faster in the middle.

## Example

With:

```python
total_steps = 1000
warmup_steps = 100
max_lr = 0.001
min_lr = 0.0001
```

the schedule gives:

| Step | Phase | Learning rate |
|---:|---|---:|
| `0` | Warmup starts | `0` |
| `50` | Halfway through warmup | `0.0005` |
| `100` | Peak / decay starts | `0.001` |
| `550` | Halfway through decay | `0.00055` |
| `1000` | Training ends | `0.0001` |
| `1100` | After training | `0.0001` |

**In short:** start small → increase linearly to `max_lr` → smoothly decrease to `min_lr` → stay there.

The intended inputs satisfy `0 <= warmup_steps < total_steps`, with a nonnegative `step` and usually `0 <= min_lr <= max_lr`. Setting `warmup_steps = 0` skips warmup. Notice that warmup starts at **zero**, not at `min_lr`.