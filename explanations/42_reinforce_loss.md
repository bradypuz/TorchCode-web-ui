This function implements the standard **REINFORCE / policy-gradient loss** for a batch of sampled actions:

\[
L = -\frac{1}{B}\sum_{i=1}^{B} r_i \log \pi_\theta(a_i \mid s_i)
\]

where:

- \(B\) is the batch size.
- `logits[i]` contains the policy’s raw scores for sample \(i\).
- \(\pi_\theta(\cdot \mid s_i)\) is the action-probability distribution obtained from those logits.
- `actions[i]` is the action that was sampled for sample \(i\).
- `rewards[i]` is the reward, return, advantage, or other scalar weight for that sampled action.

## Line-by-line explanation

```python
def reinforce_loss(logits: Tensor, actions: Tensor, rewards: Tensor) -> Tensor:
```

The inputs have these expected shapes:

- `logits`: `(B, A)`
  - `B`: number of samples or trajectories in the batch
  - `A`: number of possible actions
- `actions`: `(B,)`
  - Each entry is an integer action index in `[0, A - 1]`.
  - It should normally have dtype `torch.long`.
- `rewards`: `(B,)`
  - One scalar reward or policy-gradient weight per sampled action.

The output is a scalar tensor.

---

```python
logps = F.log_softmax(logits, dim=-1)
```

This converts raw action scores into log probabilities.

For each sample \(i\),

\[
\log p_{i,j}
=
\log
\frac{\exp(z_{i,j})}
{\sum_k \exp(z_{i,k})}
\]

where \(z_{i,j}\) is `logits[i, j]`.

If:

```python
logits.shape == (B, A)
```

then:

```python
logps.shape == (B, A)
```

`F.log_softmax` is preferable to writing:

```python
torch.log(torch.softmax(logits, dim=-1))
```

because `log_softmax` computes the result in a more numerically stable way.

---

```python
rows = torch.arange(logps.shape[0], device=logps.device)
```

This creates the batch-row indices:

```python
tensor([0, 1, 2, ..., B - 1])
```

Putting it on `logps.device` ensures it works whether `logits` is on the CPU, GPU, or another accelerator.

---

```python
chosen_logps = logps[rows, actions]
```

This selects the log probability of the action that was actually sampled for every batch element.

Conceptually, it computes:

```python
chosen_logps[i] = logps[i, actions[i]]
```

For example:

```python
actions = tensor([2, 0, 1])
```

selects:

```python
chosen_logps = tensor([
    logps[0, 2],
    logps[1, 0],
    logps[2, 1],
])
```

The resulting shape is:

```python
chosen_logps.shape == (B,)
```

An equivalent expression using `gather` is:

```python
chosen_logps = logps.gather(1, actions[:, None]).squeeze(1)
```

---

```python
return -(rewards.detach() * chosen_logps).mean()
```

This computes:

\[
-\operatorname{mean}
\left(
r_i \log \pi_\theta(a_i \mid s_i)
\right)
\]

### Why multiply by the reward?

REINFORCE uses the identity

\[
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
R \nabla_\theta
\log \pi_\theta(a \mid s)
\right]
\]

to estimate the gradient of expected reward. A sampled reward therefore weights the sampled action’s log probability.

### Why the leading minus sign?

Optimizers such as SGD and Adam perform **gradient descent**, meaning they minimize a loss. Reinforcement learning normally wants to **maximize** expected reward, so the policy objective is negated:

\[
L = -R\log\pi_\theta(a\mid s)
\]

This turns reward maximization into loss minimization.

### Why `rewards.detach()`?

`detach()` makes the rewards act as constants during backpropagation. Gradients flow through:

```python
chosen_logps -> logps -> logits
```

but not through `rewards`.

This is important when `rewards` was computed using tensors that belong to another differentiable computation graph. Without `detach()`, PyTorch could propagate the policy loss into the computation that produced the rewards, which is usually not what the score-function/REINFORCE estimator intends.

If `rewards.requires_grad` was already `False`, then `detach()` does not change the numerical result, but it documents and enforces the intended gradient behavior.

### Why `.mean()`?

The individual losses are averaged across the batch:

\[
L = -\frac{1}{B}\sum_i r_i\log\pi_\theta(a_i\mid s_i)
\]

A mean makes the approximate gradient scale less dependent on batch size. Using `.sum()` would represent the same per-sample objective but would make the gradient magnitude grow roughly with the batch size.

## What the gradient does

For batch element \(i\), let:

- \(z_{i,j}\) be the logit for action \(j\),
- \(\pi_{i,j} = \operatorname{softmax}(z_i)_j\),
- \(a_i\) be the selected action,
- \(r_i\) be the reward.

The gradient with respect to each logit is:

\[
\frac{\partial L}{\partial z_{i,j}}
=
\frac{r_i}{B}
\left(
\pi_{i,j}
-
\mathbf{1}[j=a_i]
\right)
\]

where \(\mathbf{1}[j=a_i]\) is 1 if \(j\) is the chosen action and 0 otherwise.

### Positive reward

If \(r_i > 0\), gradient descent:

- increases the chosen action’s logit,
- decreases competing actions’ logits,
- therefore makes the chosen action more probable.

### Negative reward

If \(r_i < 0\), the direction reverses:

- decreases the chosen action’s logit,
- makes the chosen action less probable.

### Zero reward

If \(r_i = 0\), that sample contributes no policy gradient.

## Small example

Suppose:

```python
logits = torch.tensor(
    [[2.0, 1.0, 0.0],
     [0.0, 1.0, 2.0]],
    requires_grad=True,
)

actions = torch.tensor([0, 1])
rewards = torch.tensor([3.0, -2.0])
```

For the first sample:

- action `0` was selected,
- its reward is positive,
- optimization will increase the probability of action `0`.

For the second sample:

- action `1` was selected,
- its reward is negative,
- optimization will decrease the probability of action `1`.

Usage:

```python
loss = reinforce_loss(logits, actions, rewards)
loss.backward()

print(loss)
print(logits.grad)
```

Only `logits` receives gradients from this loss. `actions` are integer indices and are inherently nondifferentiable, while `rewards.detach()` explicitly blocks reward gradients.

## Important conceptual distinction

This function computes the REINFORCE loss **after actions and rewards have already been obtained**. It does not sample actions itself.

A typical workflow looks like:

```python
dist = torch.distributions.Categorical(logits=logits)
actions = dist.sample()

# Interact with an environment using actions...
rewards = ...

loss = reinforce_loss(logits, actions, rewards)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The actions must actually be sampled from the policy represented by `logits`, or the estimator generally needs an off-policy correction such as an importance-sampling ratio.

## Rewards versus returns versus advantages

Although the argument is named `rewards`, in a multi-step reinforcement-learning setting it is often better to pass one of the following:

### Discounted returns

\[
G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k
\]

Then the loss for each time step is based on the future return attributable to that action.

### Advantages

\[
A_t = G_t - b(s_t)
\]

where \(b(s_t)\) is a baseline, often a learned value estimate. Advantages reduce variance while preserving the policy-gradient direction when the baseline is constructed appropriately.

For example:

```python
advantages = returns - values.detach()
loss = reinforce_loss(logits, actions, advantages)
```

Here, the behavior becomes:

- positive advantage: chosen action did better than expected, so make it more likely;
- negative advantage: chosen action did worse than expected, so make it less likely.

## Practical requirements and caveats

1. `actions` should have shape `(B,)`, integer dtype, and be on a compatible device:

   ```python
   actions.dtype == torch.long
   ```

2. Every action index must satisfy:

   ```python
   0 <= actions[i] < logits.shape[1]
   ```

3. `rewards` should be on the same device as `logits` and usually have a compatible floating-point dtype.

4. This is the basic high-variance REINFORCE estimator. Real implementations commonly add:
   - an advantage baseline,
   - return normalization,
   - entropy regularization,
   - discounting,
   - masking for variable-length trajectories.

5. `rewards.detach()` prevents gradients through reward computation, but it does not change the reward values.

In short, the function says:

> “For actions that received positive weights, increase their log probability; for actions that received negative weights, decrease their log probability; and propagate those changes only into the policy logits.”