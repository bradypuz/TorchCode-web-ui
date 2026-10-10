# Group Relative Policy Optimization Loss

Compare each response's reward to the other responses for the same prompt. Above-average responses get a positive advantage and are encouraged; below-average responses get a negative advantage and are discouraged. This reference turns those group-relative advantages into a simple policy-gradient loss.

## Inputs and output

`logps`, floating-point `rewards`, and integer `group_ids` have shape `(B,)`. Equal IDs mean responses share a prompt; IDs do not have to be consecutive or sorted. The output is a scalar tensor. Only `logps` is meant to receive gradients from this loss.

## Reading the actual code

```python
unique_ids = group_ids.unique()
advantages = torch.empty_like(rewards)
for gid in unique_ids:
    mask = group_ids == gid
    r_g = rewards[mask]
```

`unique()` finds the groups present in the batch. `empty_like` allocates storage with the rewards' shape, dtype, and device; every position is filled by its group's later assignment. Comparing IDs creates a boolean mask of shape `(B,)`. Indexing with it selects only rewards from this group, even when those positions are interleaved in the batch.

```python
mean_g = r_g.mean()
std_g = r_g.std(unbiased=False)
advantages[mask] = (r_g - mean_g) / (std_g + eps)
```

The mean provides a prompt-specific baseline. `unbiased=False` computes the population standard deviation, dividing squared deviations by the group's size. For rewards `[1,3]`, mean is `2`, squared deviations are `[1,1]`, and standard deviation is `1`. Their advantages are approximately `[-1,+1]` after dividing by `1+eps`.

For another group `[10,14]`, mean is `12` and standard deviation is `2`, so its advantages are also approximately `[-1,+1]`. A higher absolute reward baseline does not automatically make this group more influential. Epsilon is added **after** the standard deviation and prevents division by zero for equal rewards.

| Group | Rewards | Mean | Population standard deviation | Advantages with `eps=1e-5` |
| --- | --- | ---: | ---: | --- |
| 10 | `[1,3]` | 2 | 1 | approximately `[-0.999990,+0.999990]` |
| 20 | `[10,14]` | 12 | 2 | approximately `[-0.999995,+0.999995]` |
| 30 | `[7]` | 7 | 0 | `[0]` |

A singleton group has no comparison signal: its reward equals its mean, so the numerator is zero. Equal-reward multi-response groups also receive zero advantages.

```python
advantages_detached = advantages.detach()
return -(advantages_detached * logps).mean()
```

`detach()` treats reward normalization as a fixed learning signal. No gradient flows back into rewards, their group means, or their standard deviations. Multiplication is elementwise: each response's own advantage weights its log-probability. The negative mean converts a reward-maximizing objective into a loss for gradient descent.

The derivative with respect to `logps[i]` is `-advantage[i]/B`. A positive advantage gives a negative derivative, so gradient descent raises that log-probability. A negative advantage gives a positive derivative and lowers it. A zero advantage gives no signal. The scalar loss itself can be negative; that is normal for this objective.

## Run two groups and a singleton

Load the reference solution first, then run:

```python
# Example: Normalize within groups and stop gradients through rewards
logps = torch.tensor([-2.0, -1.0, -3.0, -1.0, -4.0], requires_grad=True)
rewards = torch.tensor([1.0, 3.0, 10.0, 14.0, 7.0], requires_grad=True)
group_ids = torch.tensor([10, 10, 20, 20, 30])
loss = grpo_loss(logps, rewards, group_ids)
a = 1.0 / (1.0 + 1e-5)
b = 2.0 / (2.0 + 1e-5)
torch.testing.assert_close(loss, torch.tensor(-(a + 2*b) / 5))
assert abs(loss.item() - (-0.599996)) < 1e-6
loss.backward()
torch.testing.assert_close(logps.grad, torch.tensor([a/5, -a/5, b/5, -b/5, 0.0]))
assert rewards.grad is None
```

For the first group, products are approximately `[2,-1]`; for the second, `[3,-1]`. Their sum is approximately `3`, and negating the five-response mean gives approximately `-0.6`.

## Common mistakes and reference limits

Using the default sample standard deviation changes the scale and fails for singletons. Normalizing the whole batch mixes unrelated prompts. Shapes should match exactly; accidental broadcasting changes the objective. The final mean weights responses equally, so larger groups contribute more total weight. This reference has no old-policy ratio, clipping, KL penalty, or token masking. It does not validate empty batches or epsilon.
