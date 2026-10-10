# PPO Clipped Surrogate Loss

## Intuition

Policy optimization should favor actions with positive advantages while avoiding excessively rewarding a large change from the policy that collected the data. PPO compares an ordinary importance-weighted objective with a clipped alternative and keeps the more conservative value for each example.

## Inputs and output

`new_logps`, `old_logps`, and `advantages` have shape `(B,)`. They are action or response log-probabilities and precomputed advantage estimates, not full vocabulary distributions. `clip_ratio`, default `0.2`, determines the allowed ratio interval. The output is a scalar loss tensor whose gradients flow through `new_logps`.

## Implementation walkthrough

1. Detach `old_logps` and `advantages`. The previous policy and advantage estimates act as constants for this objective.
2. Compute `ratios = exp(new_logps - old_logps_detached)`, using a log-probability difference to form the new-to-old probability ratio.
3. Form the unconstrained objective `ratios * adv_detached`.
4. Clamp ratios to `[1 - clip_ratio, 1 + clip_ratio]` and multiply by the same detached advantages.
5. Take the elementwise minimum of the two objectives, then the negative mean so minimizing the returned loss maximizes the conservative surrogate.

## Why it works

The exact implemented objective is:

$$
r_i=\exp(\log\pi_\theta(y_i)-\operatorname{stopgrad}(\log\pi_{\mathrm{old}}(y_i)))
$$

$$
\mathcal L=-\frac{1}{B}\sum_i\min\left(r_iA_i,\operatorname{clip}(r_i,1-\epsilon,1+\epsilon)A_i\right)
$$

For a positive advantage, increasing the ratio beyond the upper bound stops adding benefit. For a negative advantage, decreasing the ratio below the lower bound stops adding benefit. Moves in the harmful direction are still penalized; the algorithm does not blindly clip every gradient outside the interval. This sign-dependent behavior follows from taking the minimum **after** multiplying by the advantage.

The judge checks scalar output, a fixed numerical example, and that only current-policy log-probabilities receive gradients.

## Pitfalls and reference scope

Clamping the log-probability difference instead of the ratio changes the clipping bounds. Taking a maximum reverses the conservative choice. Detaching the new log-probabilities would remove the learning signal. This is the policy surrogate alone: it has no value loss, entropy bonus, KL penalty, advantage normalization, rollout generation, or minibatch training loop. The reference does not validate `clip_ratio`, shapes, or finite inputs; exponentiating extreme differences can overflow. Loss values need not be positive.
