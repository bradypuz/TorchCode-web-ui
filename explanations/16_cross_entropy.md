This function computes **the average cross-entropy loss for a batch of classification predictions**. It measures how much probability the model assigns to the correct classes: **higher probability for the correct class means lower loss**.

### Expected inputs

- **`logits`**: a tensor of shape `(N, C)`, where `N` is the number of examples and `C` is the number of classes. These are raw model scores—not probabilities.
- **`targets`**: an integer tensor of shape `(N,)`, containing the correct class index for each example, usually with dtype `torch.long`.

For example:

```python
logits = torch.tensor([
    [2.0, 1.0, 0.1],  # Scores for example 0
    [0.5, 2.0, 1.0],  # Scores for example 1
])
targets = torch.tensor([0, 1])  # Correct classes: 0 and 1
```

## 1. Convert scores into log-probabilities

```python
log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
```

Softmax turns a set of scores into probabilities:

\[
p_{i,c}=\frac{e^{z_{i,c}}}{\sum_j e^{z_{i,j}}}
\]

Taking its logarithm gives:

\[
\log p_{i,c}
= z_{i,c}-\log\left(\sum_j e^{z_{i,j}}\right)
\]

That is exactly what this line computes.

- **`dim=-1`**: calculate the normalization across the classes—the last dimension.
- **`keepdim=True`**: keep the result shaped `(N, 1)`, so it broadcasts correctly when subtracted from `(N, C)`.
- **`torch.logsumexp`**: computes the logarithm of the sum of exponentials in a numerically stable way, avoiding overflow from directly exponentiating large scores.

This line is equivalent to:

```python
log_probs = torch.log_softmax(logits, dim=-1)
```

## 2. Select the log-probability of each correct class

```python
log_probs[torch.arange(targets.shape[0]), targets]
```

`torch.arange(targets.shape[0])` produces the row indices:

```python
[0, 1, ..., N - 1]
```

Together with `targets`, this selects one value per example: the log-probability assigned to that example’s correct class.

For the example above, it selects:

```python
torch.stack([
    log_probs[0, 0],  # Example 0: correct class is 0
    log_probs[1, 1],  # Example 1: correct class is 1
])
```

The indexing is **pairwise**: each row index is paired with its corresponding target class.

## 3. Negate and average

```python
return -log_probs[...].mean()
```

For one example, the loss is:

\[
-\log(\text{probability assigned to the correct class})
\]

For instance:

| Probability of correct class | Loss |
|---:|---:|
| 0.9 | 0.105 |
| 0.5 | 0.693 |
| 0.1 | 2.303 |

The function averages these losses across the batch:

\[
\boxed{
L=-\frac{1}{N}\sum_{i=1}^{N}\log p_{i,\text{targets}[i]}
}
\]

All these operations support PyTorch autograd, so calling `loss.backward()` computes gradients for training.

**In practice**, for these inputs, you would normally use the built-in equivalent:

```python
loss = torch.nn.functional.cross_entropy(logits, targets)
```

Pass raw logits to either version—**do not apply softmax first**.