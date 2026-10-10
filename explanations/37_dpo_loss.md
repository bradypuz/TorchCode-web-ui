# Direct Preference Optimization Loss

DPO trains on preferred and rejected response pairs. It rewards increasing the policy's chosen-versus-rejected preference relative to a fixed reference policy. The function receives response log-probabilities that have already been computed; it does not run a language model.

## Inputs and output

`policy_chosen_logps`, `policy_rejected_logps`, `ref_chosen_logps`, and `ref_rejected_logps` each have shape `(B,)`, one score per pair. `beta` scales the comparison and defaults to `0.1`. The return is a scalar loss tensor averaged over the batch. These scores are response log-probabilities, not raw vocabulary logits.

## Reading the actual code

```python
chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
```

Subtracting logarithms forms a log probability ratio: `log(policy/reference)`. If the chosen score moves from reference `-1.5` to policy `-1`, the difference is `0.5`. This means the policy assigns that response `exp(0.5)` times the reference probability. `beta` changes the size of the training margin; the name `chosen_rewards` is an internal reference-relative score rather than an externally supplied reward.

```python
rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
```

The rejected response is evaluated in exactly the same units. With rejected policy score `-2` and reference score `-1.5`, the difference is `-0.5`. At `beta=1`, subtracting rejected reward from chosen reward gives margin `0.5 - (-0.5) = 1`.

```python
return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
```

`logsigmoid` computes the logarithm of the sigmoid stably. Negation makes the result a loss to minimize. A positive margin means the chosen response gained more relative to the reference than the rejected response did; sigmoid approaches one, and loss approaches zero. `.mean()` averages pairs rather than vocabulary tokens.

| Pair at `beta=1` | Chosen relative score | Rejected relative score | Margin | Loss |
| --- | ---: | ---: | ---: | ---: |
| Policy improves chosen preference | 0.5 | -0.5 | 1 | approximately 0.313262 |
| Policy matches reference preference | 0 | 0 | 0 | approximately 0.693147 |
| Policy reverses chosen preference | -0.5 | 0.5 | -1 | approximately 1.313262 |

The reference's default `beta=0.1` would shrink margins `1` and `-1` to `0.1` and `-0.1`. It changes both loss sensitivity and gradient scale.

## Why the signs train the desired preference

For a pair's margin `z`, the derivative of `-log(sigmoid(z))` is `sigmoid(z)-1`, which is negative. The margin contains `+beta*policy_chosen_logps` and `-beta*policy_rejected_logps`. Consequently, gradient descent raises the chosen log-probability and lowers the rejected log-probability. This describes gradients with respect to the supplied scores; model parameters receive those signals through the score computation.

At margin one and `beta=1`, a single pair's chosen derivative is approximately `-0.268941` and rejected derivative is `+0.268941`. Taking a batch mean divides each pair's contribution by `B`.

## Run loss and gradient directions

Load the reference solution first, then run:

```python
# Example: Compare easy and hard preferences and inspect gradient signs
chosen = torch.tensor([-1.0, -2.0], requires_grad=True)
rejected = torch.tensor([-2.0, -1.0], requires_grad=True)
reference = torch.tensor([-1.5, -1.5])
loss = dpo_loss(chosen, rejected, reference, reference, beta=1.0)
expected = torch.tensor((0.3132616875 + 1.3132616875) / 2)
torch.testing.assert_close(loss, expected)
assert loss.shape == ()
loss.backward()
torch.testing.assert_close(chosen.grad, torch.tensor([-0.13447071, -0.36552929]))
torch.testing.assert_close(rejected.grad, -chosen.grad)
zero_margin = dpo_loss(reference, reference, reference, reference)
torch.testing.assert_close(zero_margin, torch.tensor(0.6931471806))
```

The difficult second pair gets a larger correction because its preference margin is negative.

## Common mistakes and reference limits

Do not divide log-probabilities or apply softmax across the batch. Reversing chosen and rejected reverses the signal. The code does not detach reference inputs; callers should supply frozen reference scores when that is intended. Matching `(B,)` shapes avoids unintended broadcasting. Token summation, padding masks, response generation, and input validation happen outside this three-line loss.
