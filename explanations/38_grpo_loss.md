# Group Relative Policy Optimization Loss

## Intuition

Compare each sampled response with other responses for the same prompt. A response above its group's reward average gets a positive advantage; one below the average gets a negative advantage. Normalizing within each group prevents a prompt with a high absolute reward baseline from dominating merely because of its scale.

## Inputs and output

`logps`, `rewards`, and integer `group_ids` all have shape `(B,)`. Equal group IDs identify responses belonging to the same prompt; IDs need not be consecutive. Rewards should be floating point. The result is a scalar tensor containing an advantage-weighted policy loss.

## Implementation walkthrough

1. Find the unique group IDs and allocate an advantage tensor shaped like `rewards`.
2. For each ID, construct a boolean mask and select that group's rewards.
3. Compute the group's mean and population standard deviation with `std(unbiased=False)`.
4. Write normalized rewards `(r_g - mean_g) / (std_g + eps)` into the corresponding advantage positions.
5. Detach the finished advantage tensor. Its role is a fixed learning signal, so gradients must not flow through rewards or the normalization calculation.
6. Multiply detached advantages by policy log-probabilities and return the negative batch mean.

## Why it works

For group $g$ with $n_g$ responses:

$$
\mu_g=\frac{1}{n_g}\sum_{i\in g}r_i,\qquad\sigma_g=\sqrt{\frac{1}{n_g}\sum_{i\in g}(r_i-\mu_g)^2}
$$

$$
A_i=\frac{r_i-\mu_{g(i)}}{\sigma_{g(i)}+\epsilon},\qquad\mathcal L=-\frac{1}{B}\sum_i\operatorname{stopgrad}(A_i)\log\pi_\theta(y_i)
$$

The derivative with respect to a log-probability is `-A_i / B`: gradient descent raises probabilities for positive advantages and lowers them for negative ones. Identical-reward groups, including singleton groups, have zero advantages with positive epsilon. The judge checks numerical agreement, group independence, and that rewards receive no gradients.

## Pitfalls and reference scope

Use population standard deviation; the default sample estimate gives different scaling and is undefined for singleton groups. Add epsilon outside the standard deviation. Normalize per group rather than over the entire batch. The final mean weights individual responses equally, so larger groups contribute more total weight; it is not an equal average of group losses. This exercise implements a group-normalized REINFORCE objective without policy-ratio clipping, an old-policy term, a KL penalty, or response-token masking. A loss can be negative. The reference does not validate empty batches, shapes, or epsilon.
