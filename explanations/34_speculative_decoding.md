# Speculative Decoding Acceptance and Rejection

## Intuition

A small draft model proposes tokens cheaply. A larger target model evaluates those proposals, accepting tokens when their draft probabilities are compatible with the target. At the first rejection, a correction sample replaces that proposal and the current verification block stops.

## Inputs and output

`target_probs` and `draft_probs` have shape `(K, V)`: normalized vocabulary probabilities for each of `K` draft positions. `draft_tokens` has shape `(K,)` and contains integer token IDs. The output is a Python list containing the accepted prefix and, on rejection, one replacement token. For positive `K`, its length lies between one and `K`.

## Implementation walkthrough

1. Visit draft positions in order and extract the proposed token ID.
2. Compute target probability divided by draft probability for that token, flooring the denominator at `1e-10`.
3. Draw a uniform random number and accept if it is below the ratio capped at one. Append accepted tokens and continue.
4. At a rejection, compute the positive part of `target_probs[i] - draft_probs[i]` across the vocabulary.
5. Normalize this correction distribution. If its mass is zero, use a uniform distribution as the reference's fallback.
6. Sample and append one replacement, then return immediately. If every proposal is accepted, return all `K` draft tokens.

## Why it works

With target distribution $p$ and draft distribution $q$, acceptance uses:

$$
\alpha(v)=\min\left(1,\frac{p(v)}{q(v)}\right)
$$

A proposal drawn from $q$ contributes accepted mass $q(v)\alpha(v)=\min(q(v),p(v))$. The remaining target mass is the positive difference:

$$
r(v)=\frac{\max(p(v)-q(v),0)}{\sum_u\max(p(u)-q(u),0)}
$$

The accepted part plus the correctly weighted residual recovers the target distribution under the algorithm's sampling assumptions. This exact distribution argument ignores the reference's denominator floor: draft probabilities below `1e-10` can deviate from it, even when the two distributions are identical. The judge checks identical distributions, bounded output lengths, and valid token IDs.

## Pitfalls and reference scope

After rejection, later rows were evaluated for a prefix that no longer matches, so stop immediately. The distribution-preservation argument assumes draft tokens were sampled from their stated draft distributions and both probability tables correspond to the proper prefix. This function does not generate either table, run models, handle EOS, or append the extra target token used by some full speculative-decoding implementations after all proposals pass. Inputs are not validated, and `K=0` returns an empty list. Randomness and the denominator floor are part of the reference behavior.
