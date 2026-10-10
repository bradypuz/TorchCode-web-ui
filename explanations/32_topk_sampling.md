# Top-k and Top-p Sampling

The function chooses one next token from a vocabulary. Temperature changes how concentrated the probabilities are, top-k removes tokens below a rank cutoff, and top-p removes the tail beyond a probability-mass cutoff. The remaining probabilities are renormalized before sampling.

## Inputs and output

`logits` is a floating tensor of shape `(V,)`, with one score per token ID. The return value is one Python integer ID. `top_k=0` disables top-k and `top_p=1.0` disables top-p. This reference processes one distribution, not a batch or a complete generated sequence.

## Reading the actual code

```python
logits = logits / max(temperature, 1e-8)
```

Softmax cares about differences between logits. Dividing by temperature above one makes differences smaller and softens the distribution; a temperature between zero and one sharpens it. For logits `[0, log(4)]`, temperature one gives probabilities `[0.2, 0.8]`; temperature two gives `[1/3, 2/3]`; temperature `0.5` gives `[1/17, 16/17]`. The denominator floor makes zero and negative temperatures behave like an extremely small positive temperature. This division creates the working tensor, so later filtering does not overwrite the caller's original logits.

```python
if top_k > 0:
    top_k_val = logits.topk(top_k).values[-1]
    logits[logits < top_k_val] = float('-inf')
```

`topk` returns its largest values in descending order, so `[-1]` is the kth-largest score. Every smaller score becomes negative infinity, whose softmax probability is zero. For scores `[4, 3, 3, 1]` and `top_k=2`, the cutoff is `3`. Both tokens tied at `3` survive, so the reference can keep more than two tokens.

```python
sorted_logits, sorted_idx = torch.sort(logits, descending=True)
probs = torch.softmax(sorted_logits, dim=-1)
cumsum = torch.cumsum(probs, dim=-1)
mask = (cumsum - probs) > top_p
sorted_logits[mask] = float('-inf')
```

The nucleus filter works in sorted order. Crucially, `cumsum - probs` is the probability mass **before** each token. A token that first pushes cumulative mass over the threshold must remain: removing it could leave less than the requested mass.

For sorted probabilities `[0.5, 0.25, 0.125, 0.125]` and `top_p=0.6`:

| Sorted token | Probability | Cumulative including token | Mass before token | Masked? |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.500 | 0.500 | 0.000 | No |
| 1 | 0.250 | 0.750 | 0.500 | No; this crosses 0.6 |
| 2 | 0.125 | 0.875 | 0.750 | Yes |
| 3 | 0.125 | 1.000 | 0.875 | Yes |

The surviving mass `0.75` becomes probabilities `[2/3, 1/3, 0, 0]`. The strict `>` comparison also retains a token when its preceding mass equals the threshold. If both filters are enabled, these nucleus probabilities are computed **after top-k filtering**.

```python
logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)
probs = torch.softmax(logits, dim=-1)
return torch.multinomial(probs, 1).item()
```

`scatter_` restores vocabulary order: filtered sorted position `i` is written to original ID `sorted_idx[i]`. Otherwise sampling would return a rank instead of the correct token ID. The last softmax renormalizes the survivors. `multinomial` samples one position, and `.item()` converts its one-element tensor to a Python integer.

```torchcode-visual
sampling
```

## Run the cutoff example

Load the reference solution first, then run:

```python
# Example: Keep the token crossing the nucleus threshold
logits = torch.log(torch.tensor([0.5, 0.25, 0.125, 0.125]))
probs = torch.softmax(logits, dim=-1)
mask = (probs.cumsum(0) - probs) > 0.6
assert torch.equal(mask, torch.tensor([False, False, True, True]))
filtered = logits.clone()
filtered[mask] = float('-inf')
torch.testing.assert_close(torch.softmax(filtered, 0),
                           torch.tensor([2/3, 1/3, 0.0, 0.0]))
torch.manual_seed(7)
draws = [sample_top_k_top_p(logits, top_p=0.6) for _ in range(40)]
assert set(draws) == {0, 1}
assert sample_top_k_top_p(logits, top_k=1) == 0
```

Repeated draws can contain only the two retained IDs; top-k one makes token zero the only possible result here.

## Common mistakes and reference limits

Masking `cumsum > top_p` removes the crossing token too early. Forgetting scatter returns wrong IDs. The function does not validate `top_k <= V`, probability ranges, empty inputs, or invalid logits. Random sampling requires an explicit seed for reproducible examples.
