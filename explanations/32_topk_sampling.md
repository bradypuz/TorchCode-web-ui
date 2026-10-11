This function **randomly selects a token from a model’s output**, using three controls:

- **Temperature** adjusts how sharp or spread out the probabilities are.
- **Top-k** limits candidates to the highest-scoring tokens.
- **Top-p**, also called *nucleus sampling*, limits candidates to a group of likely tokens whose combined probability reaches a threshold.

It returns the **index of the selected token**, not the token’s text.

## 1. Adjust the logits using temperature

```python
logits = logits / max(temperature, 1e-8)
```

`logits` are the model’s raw scores for each possible token. They are not probabilities yet; higher scores mean more likely tokens.

Dividing by temperature changes the probabilities that will later be calculated:

| Temperature | Effect |
|---|---|
| `1.0` | Leaves scores unchanged |
| Between `0` and `1` | Makes sampling favor higher-scoring tokens more strongly |
| Greater than `1` | Spreads probability more evenly across tokens |

`max(temperature, 1e-8)` prevents division by zero. However, this means zero or negative temperatures are treated as a very small positive temperature—not as an explicit “pick the best token” mode.

## 2. Apply top-k filtering

```python
if top_k > 0:
    top_k_val = logits.topk(top_k).values[-1]
    logits[logits < top_k_val] = float('-inf')
```

This finds the `top_k` highest scores and uses the smallest of those as a cutoff.

For example:

```text
logits:       [5.0, 3.0, 1.0, 0.0]
top_k:        2
cutoff:       3.0
after filter: [5.0, 3.0, -inf, -inf]
```

Setting a score to negative infinity gives that token probability zero after softmax, so it cannot be sampled.

With `top_k=0`, this filtering step is skipped.

## 3. Sort the remaining scores for top-p filtering

```python
sorted_logits, sorted_idx = torch.sort(logits, descending=True)
```

This arranges tokens from highest score to lowest.

- `sorted_logits` contains the sorted scores.
- `sorted_idx` records each score’s original token index.

The indices are needed later to restore the original token order.

## 4. Keep the top-p probability group

```python
probs = torch.softmax(sorted_logits, dim=-1)
cumsum = torch.cumsum(probs, dim=-1)
mask = (cumsum - probs) > top_p
sorted_logits[mask] = float('-inf')
```

**Softmax** converts the scores into probabilities that sum to `1`.

Then `cumsum` computes the running total, starting with the most likely token.

For example, suppose the probabilities at this stage are:

```text
Sorted probabilities:     [0.50, 0.30, 0.15, 0.05]
Cumulative probabilities: [0.50, 0.80, 0.95, 1.00]
```

With `top_p=0.7`, the function keeps the first two tokens. Together, they account for `0.80` of the probability, enough to reach the threshold.

The key expression is:

```python
cumsum - probs
```

This gives the cumulative probability **before each token**:

```text
[0.00, 0.50, 0.80, 0.95]
```

Tokens whose preceding cumulative probability is greater than `top_p` are removed. This ensures that the token that takes the running total past the threshold is kept.

Unlike top-k, **top-p does not keep a fixed number of tokens**. A concentrated distribution may need only one or two; a flatter distribution may need many.

## 5. Restore the original token order

```python
logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)
```

`scatter_` places each filtered score back at its original token index.

That matters because the sampled index must refer to the original vocabulary, not to the sorted positions.

## 6. Sample from the final probabilities

```python
probs = torch.softmax(logits, dim=-1)
return torch.multinomial(probs, 1).item()
```

Softmax is applied again to **renormalize the remaining tokens’ probabilities** after filtering.

For example:

```text
Before filtering: [0.50, 0.30, 0.15, 0.05]
Remaining:        [0.50, 0.30, 0.00, 0.00]
Renormalized:     [0.625, 0.375, 0.00, 0.00]
```

`torch.multinomial(probs, 1)` randomly selects one token according to those probabilities. `.item()` converts the result into a Python integer.

## Important details

- **Top-k runs before top-p.** Top-p probabilities are calculated over the candidates that survive top-k.
- The code assumes a **one-dimensional logits tensor**, such as `[vocabulary_size]`; it is not written for batches.
- `top_k` must not exceed the number of tokens.
- Ties at the top-k cutoff can leave **more than k tokens**.
- Because the top-p mask uses `>`, if the cumulative probability equals `top_p` exactly, it also keeps the next token.

In short:

```text
Raw scores
    → temperature adjustment
    → top-k filtering
    → top-p filtering
    → final probabilities
    → randomly selected token index
```