# PPO Clipped Surrogate Loss

PPO encourages actions with positive advantages and discourages actions with negative advantages. Its clipping rule stops rewarding an excessively large change in a helpful direction, while still penalizing changes that hurt the objective. The implementation expresses this rule by comparing two values for every response.

## Inputs and output

`new_logps`, `old_logps`, and `advantages` each have shape `(B,)`. The first two are current and rollout-policy log-probabilities for the same sampled actions or responses. Advantages are supplied estimates. `clip_ratio=0.2` gives ratio bounds `[0.8,1.2]`. The return is a scalar loss tensor.

## Reading the actual code

```python
old_logps_detached = old_logps.detach()
adv_detached = advantages.detach()
```

The old policy supplied the data; it is a baseline, not a target to optimize here. Advantages also act as fixed learning signals. Detaching both prevents gradients from changing these inputs. Current `new_logps` remains differentiable.

```python
ratios = torch.exp(new_logps - old_logps_detached)
```

Subtracting logarithms and exponentiating yields `pi_new/pi_old`. If an action's probability rose from `0.2` to `0.3`, the difference is `log(1.5)` and the ratio is `1.5`. If it fell to `0.1`, the ratio is `0.5`. Equal policies give ratio one.

```python
unclipped = ratios * adv_detached
clipped = torch.clamp(ratios, 1.0 - clip_ratio,
                       1.0 + clip_ratio) * adv_detached
```

The first value is the ordinary importance-weighted objective. The second limits the ratio before multiplying by the advantage. With ratio `1.5` and advantage `+1`, these values are `1.5` and `1.2`. With advantage `-1`, they are `-1.5` and `-1.2`; the advantage's sign changes which value is smaller.

```python
return -torch.min(unclipped, clipped).mean()
```

`torch.min` is elementwise, so each sample gets the more conservative objective independently. Negation lets gradient descent maximize its mean. Taking the minimum **after multiplying by the advantage** is the key sign-dependent behavior.

| Ratio | Advantage | Unclipped | Clipped | Selected minimum | Learning behavior |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1.5 | +1 | 1.5 | 1.2 | 1.2 | No extra reward above upper bound |
| 0.5 | +1 | 0.5 | 0.8 | 0.5 | Penalize reducing a good action |
| 1.5 | -1 | -1.5 | -1.2 | -1.5 | Penalize increasing a bad action |
| 0.5 | -1 | -0.5 | -0.8 | -0.8 | No extra reward below lower bound |

For these four rows, the selected values sum to `-0.6`, their mean is `-0.15`, and the loss is `+0.15`. Rows one and four have zero gradient because the selected clipped branch is constant with respect to the ratio. Rows two and three still have a gradient despite being outside the clipping interval: their changes hurt the objective.

```torchcode-visual
ppo
```

## Run the four sign cases

Load the reference solution first, then run:

```python
# Example: Show PPO clipping depends on the advantage's sign
old_logps = torch.full((4,), 0.2).log().requires_grad_()
new_logps = torch.tensor([0.3, 0.1, 0.3, 0.1]).log().requires_grad_()
advantages = torch.tensor([1.0, 1.0, -1.0, -1.0], requires_grad=True)
loss = ppo_loss(new_logps, old_logps, advantages, clip_ratio=0.2)
torch.testing.assert_close(loss, torch.tensor(0.15))
assert loss.shape == ()
loss.backward()
torch.testing.assert_close(new_logps.grad, torch.tensor([0.0, -0.125, 0.375, 0.0]))
assert old_logps.grad is None
assert advantages.grad is None
```

The negative gradient in row two raises the good action's log-probability; the positive gradient in row three lowers the bad action's log-probability. The factor `1/4` comes from the batch mean.

## Common mistakes and reference limits

Clipping the log-probability difference uses the wrong bounds. Taking a maximum reverses the conservative rule. Detaching current scores removes learning. Shapes should all be `(B,)` to avoid accidental broadcasting. This is only the policy surrogate, with no value loss, entropy bonus, advantage normalization, rollout collection, or training loop. The code does not validate finite inputs or clip ratios; extreme differences can overflow during exponentiation. PPO loss values need not be positive.
