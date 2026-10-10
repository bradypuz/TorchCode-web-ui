# Beam Search Decoding

Beam search keeps several possible continuations alive. Each prefix carries the sum of its token log-probabilities. A prefix that starts slightly worse can eventually win if its later tokens are more likely.

## Inputs and output

`log_prob_fn` receives a one-dimensional token tensor containing the current prefix and returns next-token log-probabilities of shape `(V,)`. The other inputs are integer start/EOS IDs, a beam width, and a maximum number of expansion rounds. The output is a Python list of token IDs including the start token. Its length is at most `max_len + 1`.

## Reading the actual code

```python
beams = [(0.0, [start_token])]
completed = []
for _ in range(max_len):
    candidates = []
```

A beam is a `(score, sequence)` tuple. Score zero means no generated-token log-probabilities have been added yet; the start token itself is not scored. `completed` stores sequences that have finished, while `candidates` is rebuilt each expansion round.

```python
if seq[-1] == eos_token:
    completed.append((score, seq))
    continue
log_probs = log_prob_fn(torch.tensor(seq))
topk_lp, topk_idx = log_probs.topk(beam_width)
```

A prefix already ending in EOS moves to the completed list and gets no further tokens. Every other prefix is passed to the callback. The reference creates a fresh CPU tensor here. Each active prefix contributes its best `beam_width` next tokens; weaker local choices cannot enter the global best `beam_width` because that same prefix already has enough stronger choices.

```python
for j in range(beam_width):
    candidates.append((score + topk_lp[j].item(),
                       seq + [topk_idx[j].item()]))
candidates.sort(key=lambda x: x[0], reverse=True)
beams = candidates[:beam_width]
```

Appending to a copy of `seq` preserves the old prefix. Scores **add** because `log(P(prefix)*P(next|prefix))` equals the sum of the two logarithms. Higher scores win; for negative log-probabilities, `-1` beats `-2`. The sort combines expansions from all prefixes before pruning.

Here is a width-two example. After the start token, A has probability `0.6` and B has `0.4`. A then offers A with `0.6` and EOS with `0.4`; B offers EOS with `0.9` and B with `0.1`.

| Round-two candidate | Sequence probability | Sum of log-probabilities | Kept? |
| --- | ---: | ---: | --- |
| A, A | `0.6*0.6 = 0.36` | approximately -1.022 | Yes |
| B, EOS | `0.4*0.9 = 0.36` | approximately -1.022 | Yes |
| A, EOS | `0.6*0.4 = 0.24` | approximately -1.427 | No |
| B, B | `0.4*0.1 = 0.04` | approximately -3.219 | No |

At the next round, B/EOS becomes completed. A/A offers EOS with probability `0.9`, leaving its full sequence probability at `0.324`. B/EOS's `0.36` wins. Width one would discard B in the first round and return A/A/EOS instead.

```python
if not candidates:
    break
# After the expansion loop:
all_seqs = completed + beams
all_seqs.sort(key=lambda x: x[0], reverse=True)
return all_seqs[0][1]
```

When every active beam has ended, no expansion remains. Otherwise the loop continues until its round limit. Final ranking includes both completed sequences and remaining active beams, so a result need not end in EOS.

## Run the branching example

Load the reference solution first, then run:

```python
# Example: A weaker first token wins with a wider beam
def next_log_probs(seq):
    ids = seq.tolist()
    if len(ids) == 1:
        probs = [0.0, 0.6, 0.4, 0.0]
    elif ids[-1] == 2:
        probs = [0.0, 0.0, 0.1, 0.9]
    elif len(ids) == 2:
        probs = [0.0, 0.6, 0.0, 0.4]
    else:
        probs = [0.0, 0.1, 0.0, 0.9]
    return torch.tensor(probs).log()

assert beam_search(next_log_probs, 0, 3, 2, 3) == [0, 2, 3]
assert beam_search(next_log_probs, 0, 3, 1, 3) == [0, 1, 1, 3]
```

## Common mistakes and reference limits

The reference continues other beams after one reaches EOS; it does not immediately stop at the first finished best beam. It has no length normalization, so cumulative log-probabilities tend to favor shorter sequences. Finite-width pruning is a heuristic, and discarded paths cannot return. The code assumes a positive width no larger than the vocabulary and does not handle batching or model caches.
