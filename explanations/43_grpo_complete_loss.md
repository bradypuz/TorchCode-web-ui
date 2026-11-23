## What this function computes

`grpo_complete_loss` implements a **Group Relative Policy Optimization (GRPO)** training loss for language-model responses.

For every prompt:

1. There are `G` sampled responses.
2. Each response has a scalar reward.
3. Rewards are normalized relative to the other responses for the same prompt, producing a group-relative advantage.
4. Each response token receives that response’s advantage.
5. A PPO-style clipped objective updates the policy using the ratio between the current policy and the rollout policy.
6. A KL penalty discourages the current policy from moving too far from a fixed reference policy.
7. Padding tokens are ignored.
8. Token losses are averaged within each response, then responses are averaged together.

The overall structure is:

\[
\text{loss}
=
\frac{1}{BG}
\sum_{b=1}^{B}
\sum_{g=1}^{G}
\frac{1}{N_{b,g}}
\sum_{t \in \text{valid}(b,g)}
\left[
-\min\left(r_{b,g,t}A_{b,g},
\operatorname{clip}(r_{b,g,t})A_{b,g}\right)
+
\beta K_{b,g,t}
\right]
\]

where:

- \(A_{b,g}\) is a reward-derived group-relative advantage,
- \(r_{b,g,t}\) is the current-to-old policy probability ratio,
- \(K_{b,g,t}\) is the sampled-token KL penalty,
- \(N_{b,g}\) is the response’s number of valid tokens.

---

# Inputs and shapes

```python
policy_logps: Tensor  # (B, G, T)
old_logps: Tensor     # (B, G, T)
ref_logps: Tensor     # (B, G, T)
rewards: Tensor       # (B, G)
token_mask: Tensor    # (B, G, T)
```

The dimensions mean:

- `B`: number of prompts in the batch.
- `G`: number of sampled responses per prompt.
- `T`: padded response length.

For example, if:

```text
B = 8 prompts
G = 16 responses per prompt
T = 512 token positions
```

then each log-probability tensor has shape `(8, 16, 512)`.

The three log-probability tensors refer to the **same sampled tokens**, evaluated under different policies:

- `policy_logps`: log-probabilities under the policy currently being trained.
- `old_logps`: log-probabilities under the policy used to generate the rollouts.
- `ref_logps`: log-probabilities under a fixed reference policy.

If sampled token \(y_{b,g,t}\) appears at a position, then conceptually:

\[
\text{policy\_logps}_{b,g,t}
=
\log \pi_\theta(y_{b,g,t}\mid x_b,y_{b,g,<t})
\]

and similarly for the old and reference policies.

---

# 1. Sanitize padding

```python
valid = token_mask.detach().bool()
current = torch.where(valid, policy_logps, 0.0)
old = torch.where(valid, old_logps.detach(), 0.0)
ref = torch.where(valid, ref_logps.detach(), 0.0)
```

## Why the mask is detached

```python
valid = token_mask.detach().bool()
```

The mask is treated as fixed metadata, not something differentiable. It determines which positions belong to real response tokens.

The mask generally looks like:

```text
[1, 1, 1, 1, 0, 0, 0]
```

where:

- `1` or `True`: a real response token,
- `0` or `False`: padding.

## Why `torch.where` is used before arithmetic

Padding entries in the input tensors might contain `NaN`, `Inf`, or arbitrary values. If the implementation directly computed:

```python
torch.exp(policy_logps - old_logps)
```

then an invalid padded entry such as:

```text
NaN - NaN
```

would remain `NaN`. Multiplying it by zero later would not fix it:

\[
0 \cdot \mathrm{NaN} = \mathrm{NaN}.
\]

Instead, invalid positions are replaced with zero before subtraction and exponentiation:

```python
current = torch.where(valid, policy_logps, 0.0)
```

Thus, at a padding position:

```text
current = 0
old = 0
ref = 0
```

and the subsequent calculations stay finite.

## Gradient behavior

Only `policy_logps` is intended to receive gradients.

- `policy_logps` is not detached.
- `old_logps` is detached.
- `ref_logps` is detached.
- `token_mask` is detached.

At invalid positions, `torch.where` selects the constant `0.0`, so the derivative with respect to `policy_logps` is zero there.

Therefore gradients flow only through valid entries of `policy_logps`.

---

# 2. Compute group-relative advantages

```python
mean = rewards.mean(dim=1, keepdim=True)
std = rewards.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
advantages = ((rewards - mean) / std).detach().unsqueeze(-1)
```

GRPO does not require a separately trained value model in the way ordinary actor-critic PPO often does. Instead, it compares responses sampled for the same prompt.

For prompt \(b\), let its `G` response rewards be:

\[
R_{b,1}, R_{b,2}, \ldots, R_{b,G}.
\]

The function computes:

\[
\mu_b
=
\frac{1}{G}\sum_{g=1}^{G}R_{b,g}
\]

and the population standard deviation:

\[
\sigma_b
=
\sqrt{
\frac{1}{G}
\sum_{g=1}^{G}
(R_{b,g}-\mu_b)^2
}.
\]

Then the advantage is:

\[
A_{b,g}
=
\frac{R_{b,g}-\mu_b}
{\max(\sigma_b,10^{-6})}.
\]

This is done independently for each prompt.

## Example

Suppose one prompt has four responses with rewards:

```text
[1, 2, 4, 5]
```

Their mean is:

\[
\mu = 3.
\]

Their advantages are normalized versions of:

```text
[-2, -1, +1, +2]
```

Therefore:

- responses scoring above the group average get positive advantages,
- responses scoring below the group average get negative advantages.

The absolute scale of the reward is mostly removed. For example, changing rewards from:

```text
[1, 2, 4, 5]
```

to:

```text
[10, 20, 40, 50]
```

produces the same normalized advantages, apart from numerical details.

## Why `unbiased=False`

```python
rewards.std(..., unbiased=False)
```

This computes the population standard deviation, dividing by `G`, rather than the sample estimate dividing by `G - 1`.

That is appropriate here because the goal is group normalization, not statistical estimation of an unknown population variance. It also avoids the `G = 1` problem associated with the unbiased standard deviation.

## Why clamp the standard deviation

```python
.clamp_min(1e-6)
```

If all responses receive the same reward, their standard deviation is zero. Dividing by zero would generate `NaN`.

With the clamp:

```text
rewards = [3, 3, 3, 3]
mean    = 3
numerator = [0, 0, 0, 0]
```

so every advantage is zero:

```text
advantages = [0, 0, 0, 0]
```

In that situation, the policy-gradient part contributes no learning signal, although the KL penalty can still contribute.

## Why detach the advantages

```python
advantages = (...).detach()
```

The rewards and their normalization are treated as fixed targets. No gradient flows into the reward computation or reward model through this loss.

## Why `unsqueeze(-1)`

Before unsqueezing:

```text
advantages.shape == (B, G)
```

After:

```text
advantages.shape == (B, G, 1)
```

This allows the same response-level advantage to broadcast over all `T` tokens:

\[
A_{b,g,1}
=
A_{b,g,2}
=
\cdots
=
A_{b,g,T}
=
A_{b,g}.
\]

Every token in a response is therefore reinforced or discouraged according to the response’s scalar outcome.

---

# 3. Compute the current-to-old probability ratio

```python
ratio = torch.exp(current - old)
```

Because:

\[
\exp(\log a-\log b)=\frac{a}{b},
\]

the ratio is:

\[
r_{b,g,t}
=
\frac{
\pi_\theta(y_{b,g,t}\mid x_b,y_{b,g,<t})
}{
\pi_{\text{old}}(y_{b,g,t}\mid x_b,y_{b,g,<t})
}.
\]

Interpretation:

- `ratio == 1`: current policy gives the sampled token the same probability as the rollout policy.
- `ratio > 1`: current policy makes that token more likely.
- `ratio < 1`: current policy makes that token less likely.

Using log-probability differences is the normal numerically stable way to calculate probability ratios.

This is a **per-token** ratio, not a full-response product of token ratios.

---

# 4. Build the PPO-style clipped objective

```python
unclipped = ratio * advantages
clipped = ratio.clamp(1 - epsilon, 1 + epsilon) * advantages
policy_loss = -torch.minimum(unclipped, clipped)
```

With the default:

```python
epsilon = 0.2
```

the clipped ratio lies in:

\[
[1-\epsilon,1+\epsilon]=[0.8,1.2].
\]

The surrogate objective being maximized is:

\[
\min
\left(
rA,
\operatorname{clip}(r,1-\epsilon,1+\epsilon)A
\right).
\]

Because PyTorch optimizers normally **minimize** a loss, the code negates it:

\[
L_{\text{policy}}
=
-
\min
\left(
rA,
\operatorname{clip}(r,1-\epsilon,1+\epsilon)A
\right).
\]

## Why clipping is useful

Without clipping, a policy update could greatly increase or decrease the probability of a sampled token, causing unstable training.

The clipping creates a conservative or pessimistic objective: the optimizer cannot claim an arbitrarily large improvement merely by moving the ratio far outside the trusted interval.

---

## Positive-advantage case

Suppose:

\[
A > 0.
\]

This means the response was better than its peers, so its sampled tokens should generally become more likely.

If:

```text
ratio = 1.1
```

then this is inside the clipping interval, so the ordinary ratio is used.

If:

```text
ratio = 2.0
```

then:

\[
rA=2A,
\qquad
\operatorname{clip}(r)A=1.2A.
\]

The minimum is:

\[
1.2A.
\]

Thus the objective stops rewarding further probability increases beyond the upper clipping boundary.

For positive advantages, clipping mainly limits overly large probability increases.

---

## Negative-advantage case

Suppose:

\[
A < 0.
\]

This means the response was worse than its peers, so its sampled tokens should generally become less likely.

Consider:

```text
A = -1
ratio = 0.5
```

Then:

\[
rA = -0.5
\]

and:

\[
\operatorname{clip}(r)A
=
0.8(-1)
=
-0.8.
\]

The minimum is:

\[
-0.8,
\]

not \(-0.5\). This prevents the objective from receiving an excessive apparent improvement by driving a bad token’s probability too far downward.

For negative advantages, clipping mainly limits overly large probability decreases.

## Why `minimum` is correct for both signs

A common source of mistakes is to reason that negative advantages require changing `minimum` to `maximum`. They do not.

The standard PPO objective is always:

\[
\min(rA,\operatorname{clip}(r)A).
\]

Multiplying by a negative advantage automatically reverses the ordering in the appropriate way.

---

# 5. Compute the reference-policy KL penalty

```python
log_ratio = ref - current
kl = torch.expm1(log_ratio) - log_ratio
```

Let:

\[
x
=
\log \pi_{\text{ref}}(y)
-
\log \pi_\theta(y)
=
\log\frac{\pi_{\text{ref}}(y)}{\pi_\theta(y)}.
\]

The code computes:

\[
K(x)
=
e^x-1-x.
\]

Equivalently:

\[
K
=
\frac{\pi_{\text{ref}}(y)}
{\pi_\theta(y)}
-
1
-
\log
\frac{\pi_{\text{ref}}(y)}
{\pi_\theta(y)}.
\]

Or, rearranging the logarithm:

\[
K
=
\log
\frac{\pi_\theta(y)}
{\pi_{\text{ref}}(y)}
+
\frac{\pi_{\text{ref}}(y)}
{\pi_\theta(y)}
-1.
\]

This is commonly called the sampled-token or Schulman-style KL estimator.

## Why it behaves like a penalty

For every real number \(x\):

\[
e^x \geq 1+x.
\]

Therefore:

\[
e^x-1-x\geq0.
\]

So `kl` is pointwise nonnegative, up to floating-point rounding, and is zero when:

\[
x=0
\quad\Longleftrightarrow\quad
\pi_\theta(y)=\pi_{\text{ref}}(y).
\]

This is useful because a simpler sampled term such as:

\[
\log\pi_\theta(y)-\log\pi_{\text{ref}}(y)
\]

can be negative for an individual sampled token even though the exact KL divergence is nonnegative.

## Relationship to the true KL

If \(y\) is sampled from the current policy \(p=\pi_\theta\), and \(q=\pi_{\text{ref}}\), then:

\[
\mathbb{E}_{y\sim p}
\left[
\frac{q(y)}{p(y)}
-1
-\log\frac{q(y)}{p(y)}
\right]
=
D_{\mathrm{KL}}(p\|q).
\]

To see this:

\[
\begin{aligned}
\mathbb{E}_{p}[K]
&=
\sum_y p(y)
\left[
\frac{q(y)}{p(y)}
-1
-\log\frac{q(y)}{p(y)}
\right]\\
&=
\sum_y q(y)
-\sum_y p(y)
+
\sum_y p(y)\log\frac{p(y)}{q(y)}\\
&=
1-1+D_{\mathrm{KL}}(p\|q)\\
&=
D_{\mathrm{KL}}(p\|q).
\end{aligned}
\]

Thus the direction is conceptually:

\[
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\text{ref}}).
\]

In this function, however, tokens were originally generated by the old rollout policy rather than necessarily by the exact current policy. Consequently, once the current policy has moved away from the rollout policy, the finite-batch quantity should be understood as the usual sampled-token KL proxy used in GRPO/PPO training, rather than an exact unbiased current-policy KL estimate.

## Why use `expm1`

```python
torch.expm1(log_ratio)
```

computes:

\[
e^x-1
\]

directly.

For small \(x\), calculating:

```python
torch.exp(x) - 1
```

can lose precision because `exp(x)` is extremely close to `1`, and subtracting two nearly equal numbers causes cancellation.

`expm1(x)` is designed to compute \(e^x-1\) accurately near zero.

For small \(x\), the KL term behaves like:

\[
e^x-1-x
=
\frac{x^2}{2}
+
\frac{x^3}{6}
+\cdots.
\]

So near the reference policy, it is approximately quadratic:

\[
K(x)\approx\frac{x^2}{2}.
\]

---

# 6. Combine the policy loss and KL penalty

```python
token_losses = policy_loss + beta * kl
```

With the default:

```python
beta = 0.04
```

each valid token gets:

\[
L_{b,g,t}
=
L_{\text{policy},b,g,t}
+
0.04\,K_{b,g,t}.
\]

The two parts have different purposes:

- `policy_loss` moves the policy toward responses with positive relative advantage and away from responses with negative relative advantage.
- `beta * kl` pulls the policy toward the reference distribution and limits drift.

Larger `beta` means stronger reference regularization:

- `beta = 0`: no reference-policy penalty.
- small `beta`: more freedom to optimize rewards.
- large `beta`: stronger resistance to moving away from the reference model.

This KL penalty is separate from PPO clipping:

- PPO clipping constrains movement relative to the **old rollout policy**.
- KL regularization constrains movement relative to the **reference policy**.

The old and reference policies can be different:

- the old policy is usually a recent policy snapshot used to generate the batch,
- the reference policy is often a frozen pretrained or supervised-fine-tuned model.

---

# 7. Mask padding and average within each response

```python
mask = valid.to(policy_logps.dtype)
per_response_loss = (
    (token_losses * mask).sum(dim=-1)
    / mask.sum(dim=-1).clamp_min(1)
)
```

The Boolean mask is converted to the policy’s floating-point dtype:

```text
True  -> 1
False -> 0
```

For a particular response:

\[
L_{b,g}
=
\frac{
\sum_{t=1}^{T}
m_{b,g,t}L_{b,g,t}
}{
\max\left(
\sum_{t=1}^{T}m_{b,g,t},
1
\right)
}.
\]

This is the mean loss over the valid response tokens.

## Why average instead of sum

Suppose one response contains 10 tokens and another contains 100 tokens.

If token losses were simply summed, the 100-token response could contribute approximately ten times as much to the final loss solely because it is longer.

Averaging within each response gives both responses equal total weight:

```text
response A: average of its 10 token losses
response B: average of its 100 token losses
```

Then each response contributes one scalar to the final mean.

This means individual tokens in shorter responses have more weight than individual tokens in longer responses. That is intentional under equal-response weighting.

## Why clamp the token count

```python
mask.sum(dim=-1).clamp_min(1)
```

Normally every response should have at least one valid token. But if a response is completely empty, its token count is zero.

The clamp prevents division by zero:

- numerator: zero because every mask entry is zero,
- denominator: clamped to one,
- per-response loss: zero.

One subtle consequence is that an empty response still occupies a slot in the final mean and contributes zero, reducing the overall scale slightly. In a well-formed batch, responses should generally not be completely empty.

---

# 8. Average across all prompts and responses

```python
return per_response_loss.mean()
```

At this point:

```text
per_response_loss.shape == (B, G)
```

The final result is:

\[
L
=
\frac{1}{BG}
\sum_{b=1}^{B}
\sum_{g=1}^{G}
L_{b,g}.
\]

It is a scalar suitable for:

```python
loss.backward()
optimizer.step()
```

Because every prompt has the same number `G` of responses, this also gives each prompt equal total weight.

---

# Full conceptual data flow

The function can be summarized as:

```text
rewards (B, G)
    │
    ├─ group mean/std per prompt
    │
    └─ normalized advantages (B, G, 1)
                           │
policy_logps ─┐            │
old_logps ────┴─ ratio ────┴─ PPO clipped token objective
                                              │
policy_logps ─┐                               │
ref_logps ────┴─ sampled KL penalty ──────────┤
                                              │
                                  combined token losses
                                              │
                                      apply token mask
                                              │
                               mean over tokens per response
                                              │
                               mean over prompts/responses
                                              │
                                          scalar loss
```

---

# Exactly where gradients flow

The intended trainable input is `policy_logps`.

## Gradients do flow through

- valid entries of `policy_logps`,
- the probability ratio,
- the unclipped/clipped PPO objective where active,
- the reference KL term.

## Gradients do not flow through

- `old_logps`, because of `.detach()`,
- `ref_logps`, because of `.detach()`,
- rewards and normalized advantages, because `advantages` is detached,
- `token_mask`, because it is detached,
- invalid/padded entries of `policy_logps`, because `torch.where` selects zero there.

In symbolic terms:

\[
\frac{\partial L}
{\partial \text{policy\_logps}_{b,g,t}}
=0
\quad\text{when}\quad
m_{b,g,t}=0.
\]

---

# Important edge cases

## All rewards for a prompt are equal

Then:

```text
advantages = 0
```

for that prompt.

The PPO/GRPO reward objective contributes no gradient there. The KL penalty may still move the policy toward the reference policy.

## `G = 1`

With only one response per prompt:

```text
reward == group mean
advantage == 0
```

so there is no relative reward signal. Only the KL penalty remains.

GRPO therefore relies on having multiple responses per prompt.

## A response has no valid tokens

Its per-response loss becomes zero because of the denominator clamp. It still appears in the final average.

## Padding contains `NaN` or `Inf`

It is safely replaced with zero before exponentiation and subtraction, assuming the mask correctly marks those entries invalid.

## Valid positions contain extreme log-probability differences

`torch.exp(current - old)` and `torch.expm1(ref - current)` can still overflow for sufficiently extreme valid values. The sanitization only protects padding. In normal training, valid log-probability differences should generally remain in a manageable range, aided by frequent rollout refreshes, clipping, and KL regularization.

## Non-finite rewards

The function does not sanitize rewards. A `NaN` or `Inf` reward can contaminate the group mean, standard deviation, and all advantages for that prompt.

---

# A compact mathematical restatement

For each prompt \(b\) and response \(g\):

### Group-relative advantage

\[
A_{b,g}
=
\frac{R_{b,g}-\mu_b}
{\max(\sigma_b,10^{-6})}.
\]

### Token probability ratio

\[
r_{b,g,t}
=
\exp
\left(
\log\pi_\theta(y_{b,g,t})
-
\log\pi_{\text{old}}(y_{b,g,t})
\right).
\]

### Clipped policy loss

\[
L^{\text{policy}}_{b,g,t}
=
-
\min
\left[
r_{b,g,t}A_{b,g},
\operatorname{clip}
(r_{b,g,t},1-\epsilon,1+\epsilon)
A_{b,g}
\right].
\]

### Sampled KL penalty

\[
x_{b,g,t}
=
\log\pi_{\text{ref}}(y_{b,g,t})
-
\log\pi_\theta(y_{b,g,t}),
\]

\[
K_{b,g,t}
=
e^{x_{b,g,t}}-1-x_{b,g,t}.
\]

### Combined token loss

\[
L^{\text{token}}_{b,g,t}
=
L^{\text{policy}}_{b,g,t}
+
\beta K_{b,g,t}.
\]

### Response and batch reduction

\[
L
=
\frac{1}{BG}
\sum_{b,g}
\frac{
\sum_t m_{b,g,t}L^{\text{token}}_{b,g,t}
}{
\max(\sum_t m_{b,g,t},1)
}.
\]

In short, the function says:

> For each prompt, increase the likelihood of tokens from better-than-average responses and decrease the likelihood of tokens from worse-than-average responses, but clip aggressive changes relative to the rollout policy, penalize drift from the reference policy, ignore padding safely, and weight each response equally regardless of length.