# Beam Search Decoding

## Intuition

Greedy decoding can pick a promising token that leads to a poor sequence. Beam search keeps several candidate prefixes so a slightly weaker early choice can win after later probabilities are considered. Each candidate carries the sum of its token log-probabilities.

## Inputs and output

`log_prob_fn` receives the current sequence as a one-dimensional token tensor and returns `(V,)` next-token log-probabilities. `start_token` and `eos_token` are integer IDs. `beam_width` controls the number of active candidates. `max_len` limits expansion rounds after the start token, so returned length can be up to `max_len + 1`. The result is a Python list of integer IDs including the start token.

## Implementation walkthrough

1. Initialize a single beam `(0.0, [start_token])` and a separate completed-sequence list.
2. In each round, move beams already ending in EOS to the completed list without extending them.
3. For every other beam, ask the callback for next-token log-probabilities and expand only its top `beam_width` tokens.
4. Add each new log-probability to the prefix score. Sort all expansions together and retain the highest-scoring `beam_width` candidates.
5. Stop when no expansions remain or the loop reaches `max_len`. Rank completed sequences together with the remaining beams and return the best sequence.

## Why it works

Multiplication of conditional token probabilities becomes addition in log space:

$$
\log P(y_1,\ldots,y_L\mid y_0)=\sum_{t=1}^{L}\log P(y_t\mid y_0,\ldots,y_{t-1})
$$

Comparing cumulative scores lets multiple early paths compete. Keeping each prefix's best `beam_width` extensions is sufficient for selecting that many candidates from the combined extensions. The judge includes a case where width two finds a better completed path than greedy decoding.

## Pitfalls and reference scope

The actual reference does **not** stop immediately when the best beam reaches EOS, despite that wording in the template: other active beams continue expanding. A final active sequence may win even if it has not reached EOS. Scores have no length normalization, so raw log-probabilities tend to favor shorter sequences. This is a finite-width heuristic, not a guarantee of the globally most likely sequence. The callback receives a newly created CPU tensor. The reference has no device option, positive-width validation, vocabulary-size guard, batching, or model-cache management.
