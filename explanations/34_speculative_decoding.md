# Speculative Decoding Acceptance and Rejection

A draft model proposes a short block of tokens. This function checks those proposals against target-model probabilities, keeps an accepted prefix, and replaces the first rejected token. It then stops because later probability rows belong to the original proposed prefix.

## Inputs and output

`target_probs` and `draft_probs` have shape `(K, V)`, with a normalized vocabulary distribution at each draft position. `draft_tokens` is an integer tensor of shape `(K,)`. The output is a Python list. If all proposals pass it contains `K` tokens; otherwise it contains accepted earlier tokens followed by one correction token.

## Reading the actual code

```python
K = len(draft_tokens)
accepted = []
for i in range(K):
    t = draft_tokens[i].item()
    ratio = target_probs[i, t] / max(draft_probs[i, t].item(), 1e-10)
```

Positions are processed in order. `.item()` turns the proposed ID into a Python integer so it can index the row. The ratio compares target probability `p(t)` to draft probability `q(t)`. If the target likes the proposal at least as much as the draft does, its ordinary acceptance probability is one. If `p(t)=0.2` and `q(t)=0.5`, the ratio is `0.4`.

```python
if torch.rand(1).item() < min(1.0, ratio.item()):
    accepted.append(t)
```

The random draw is uniform between zero and one. Capping the ratio at one makes it a probability. A draw of `0.3` accepts the `0.4` example, while `0.7` rejects it. Accepted IDs are appended unchanged, and the loop proceeds to the next position.

| Proposed token probability | Draft `q(t)` | Target `p(t)` | Acceptance probability |
| --- | ---: | ---: | ---: |
| Target prefers it | 0.2 | 0.5 | 1 |
| Target prefers it less | 0.5 | 0.2 | 0.4 |
| Target forbids it | 1 | 0 | 0 |
| Identical tiny probability | `1e-12` | `1e-12` | 0.01 because denominator is floored |

The last row is a reference-specific detail. The denominator becomes `1e-10`, so identical distributions do not guarantee acceptance for an extremely rare supplied proposal. This differs from the ideal `p/q` ratio, which would be one.

```python
adjusted = torch.clamp(target_probs[i] - draft_probs[i], min=0)
s = adjusted.sum()
if s > 0:
    adjusted = adjusted / s
else:
    adjusted = torch.ones_like(adjusted) / adjusted.shape[0]
```

After rejection, sample where the target needs more mass than the draft supplied. For `p=[0.2,0.8]` and `q=[0.5,0.5]`, the difference is `[-0.3,0.3]`, the positive part is `[0,0.3]`, and normalization gives `[0,1]`. Token one must replace a rejected token zero. If every positive difference is zero, the code falls back to uniform probabilities. That fallback can be reached when the denominator floor causes rejection with identical distributions.

```python
accepted.append(torch.multinomial(adjusted, 1).item())
return accepted
```

The immediate return matters: a replacement changes the prefix, making later rows inappropriate. If the loop finishes without rejection, the final return simply gives the full accepted block.

## Run acceptance, rejection, and the tiny-probability case

Load the reference solution first, then run:

```python
# Example: Verify accepted blocks, correction sampling, and the denominator floor
torch.manual_seed(0)
equal = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
assert speculative_decode(equal, equal, torch.tensor([0, 1])) == [0, 1]

target = torch.tensor([[0.0, 1.0], [0.5, 0.5]])
draft = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
assert speculative_decode(target, draft, torch.tensor([0, 0])) == [1]

tiny = torch.tensor([[1e-12, 1.0 - 1e-12], [0.5, 0.5]], dtype=torch.float64)
ratio = tiny[0, 0] / max(tiny[0, 0].item(), 1e-10)
assert abs(ratio.item() - 0.01) < 1e-12
torch.manual_seed(0)
assert torch.rand(1).item() > ratio.item()
torch.manual_seed(0)
result = speculative_decode(tiny, tiny, torch.tensor([0, 1]))
assert len(result) == 1 and result[0] in (0, 1)
```

The second call replaces a guaranteed rejection and skips position one. The last call exposes the floor rather than silently assuming identical rows always pass.

## Common mistakes and reference limits

The target-distribution argument assumes draft tokens were sampled from their stated distributions and every row matches its prefix; the floor slightly changes that ideal argument. This function does not run either model, generate proposals, handle EOS, or append an extra target token after full acceptance. It does not validate inputs, and `K=0` returns an empty list.
