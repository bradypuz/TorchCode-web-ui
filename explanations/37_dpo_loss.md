# Direct Preference Optimization Loss

## Intuition

Each training example contains a preferred response and a rejected response. DPO asks the current policy to increase its preference for the chosen response relative to a reference policy. The reference-relative comparison focuses on how the policy changes the pair's odds rather than the raw likelihood of either response alone.

## Inputs and output

All four inputs have shape `(B,)`: policy and reference log-probabilities for chosen and rejected responses. These are already-computed response scores, not vocabulary logits. `beta` is a scalar scale, defaulting to `0.1`. The result is one scalar tensor, averaged across preference pairs.

## Implementation walkthrough

1. Compute `chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)`. Subtracting log-probabilities gives the logarithm of a policy-to-reference probability ratio.
2. Compute the same reference-relative score for rejected responses.
3. Subtract rejected scores from chosen scores to obtain a per-pair preference margin.
4. Apply `F.logsigmoid`, negate, and take the batch mean. The functional implementation evaluates the log-sigmoid stably without explicitly forming a potentially tiny sigmoid value.

## Why it works

For chosen response $y_w$ and rejected response $y_l$, define:

$$
z=\beta\left[\log\pi_\theta(y_w)-\log\pi_{\mathrm{ref}}(y_w)-\log\pi_\theta(y_l)+\log\pi_{\mathrm{ref}}(y_l)\right]
$$

$$
\mathcal L=-\frac{1}{B}\sum_{i=1}^{B}\log\sigma(z_i)
$$

A large positive margin yields a small loss. A negative margin yields a larger penalty. When the policy and reference have the same chosen-versus-rejected preference, the margin is zero and the per-pair loss is $\log 2$. Its derivative with respect to the margin is $\sigma(z)-1$, so gradient descent encourages a larger chosen-relative advantage. Beta controls the scale of this comparison and its gradients.

The judge checks easy and hard pairs, agreement with the formula, gradients to both policy scores, and the effect of changing beta.

## Pitfalls and reference scope

Subtract log-probabilities; do not divide them or apply softmax across the batch. Reversing the chosen/rejected subtraction reverses the learning signal. The reference does not detach reference log-probabilities internally: callers should provide frozen reference scores when that is intended. It does not compute token masks, sum token log-probabilities, sample preference pairs, or validate input shapes and beta. Those steps belong outside this scalar objective.
