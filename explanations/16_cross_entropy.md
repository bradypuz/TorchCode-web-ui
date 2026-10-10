# Cross-entropy loss

## What this function computes

A classifier produces one score for each possible class. These scores, called **logits**, do not have to be positive or add up to one. Cross-entropy measures how much probability the classifier assigns to the correct class: a high correct-class probability gives a small loss; a low one gives a large loss.

Here `logits` has shape `(N, C)`: `N` examples and `C` classes. `targets` is an integer tensor of shape `(N,)`; each entry is a class index from `0` through `C - 1`. The return value is one scalar tensor, the mean loss across the examples. Targets are indices, not vectors of class probabilities.

## Convert scores to log probabilities safely

```python
log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
```

For one row, softmax would compute `exp(score) / sum(exp(all_scores))`. Taking its logarithm gives `score - log(sum(exp(all_scores)))`, which is exactly the expression above. Working directly with logarithms avoids forming a potentially enormous exponential and then taking its logarithm.

`torch.logsumexp` performs that calculation safely. Internally, subtracting a row's largest score keeps the exponentials small. For logits `[1000, 1001]`, naïvely exponentiating both entries overflows ordinary floating-point numbers. `logsumexp` still produces meaningful log probabilities.

`dim=-1` means sum over classes. `keepdim=True` keeps the result shaped `(N, 1)`. When this is subtracted from `(N, C)`, PyTorch repeats each row's one normalizing value across its classes. This is **broadcasting**, matching dimensions of size one without manually copying them.

## Select the correct class from each row

```python
return -log_probs[torch.arange(targets.shape[0]), targets].mean()
```

For three targets `[2, 0, 1]`, `torch.arange(3)` gives row indices `[0, 1, 2]`. Together these select `log_probs[0, 2]`, `log_probs[1, 0]`, and `log_probs[2, 1]`. They select three paired coordinates, not a rectangular submatrix.

The selected log probabilities are usually negative. The minus sign turns them into nonnegative losses, and `.mean()` averages those losses. The scalar remains connected to the logits, so calling `.backward()` computes gradients.

## Two examples, including a large score offset

Use logits equal to the logarithms of these already-normalized probabilities. Their `logsumexp` is zero, making each intermediate value easy to inspect.

| Example | Class probabilities | Target | Selected log probability | Individual loss |
|---|---|---|---|---|
| 0 | `[0.25, 0.75]` | `1` | `log(0.75) ≈ -0.2877` | `0.2877` |
| 1 | `[0.80, 0.20]` | `0` | `log(0.80) ≈ -0.2231` | `0.2231` |

Their mean is about `0.2554`. Adding `1000` to every score in a row leaves softmax probabilities unchanged: the common exponential factor cancels. The second assertion below checks this stability property.

```python
# Example: Average correct-class losses and ignore a common score offset
probabilities = torch.tensor([[0.25, 0.75], [0.80, 0.20]])
logits = probabilities.log().detach().requires_grad_(True)
targets = torch.tensor([1, 0], dtype=torch.long)
loss = cross_entropy_loss(logits, targets)
expected = -(torch.log(torch.tensor(0.75)) + torch.log(torch.tensor(0.80))) / 2
assert loss.shape == torch.Size([])
assert torch.allclose(loss, expected, atol=1e-6)
assert torch.allclose(cross_entropy_loss(logits + 1000, targets), expected, atol=1e-4)
loss.backward()
assert torch.allclose(logits.grad, torch.tensor([[0.125, -0.125], [-0.10, 0.10]]), atol=1e-6)
```

The negative gradient at a correct-class score says increasing that score decreases the loss. The positive gradient at the other score says increasing it increases the loss. Division by two comes from averaging two examples.

## Common mistakes and reference scope

Do not apply softmax before passing logits here, and do not take the log of raw logits. Keep the normalizing dimension so row subtraction broadcasts correctly. This implementation expects a nonempty 2-D batch with valid target indices. It does not implement label smoothing, ignored targets, class weighting, or a selectable reduction.
