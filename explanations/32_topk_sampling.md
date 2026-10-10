# Top-k and Top-p Sampling

## Intuition

Sampling keeps generation varied while filtering unlikely tokens. Temperature changes how sharply the distribution favors high logits. Top-k keeps a fixed group of strong candidates; top-p keeps a prefix covering a chosen probability mass. This reference applies both filters, then draws one token.

## Inputs and output

`logits` is a floating tensor of shape `(V,)`, one score per vocabulary token. `top_k=0` disables the top-k filter and `top_p=1.0` disables the top-p filter. The output is one Python integer token index. This function handles a single sampling step, without a sequence, length limit, or EOS stopping rule.

## Implementation walkthrough

1. Divide logits by `max(temperature, 1e-8)`. The division creates the working tensor; lower positive temperatures make softmax more concentrated.
2. If `top_k > 0`, find the kth-largest logit and replace scores strictly below it with negative infinity.
3. If `top_p < 1.0`, sort the surviving scores in descending order, compute softmax probabilities, and take their cumulative sum.
4. Mask a token only when the probability mass strictly before it exceeds `top_p`. This keeps the token that crosses the threshold.
5. Scatter sorted, filtered scores back to their original vocabulary positions. Apply softmax and call `torch.multinomial` to sample one index.

## Why it works

Before filtering, temperature gives:

$$
p_i=\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}
$$

Setting a score to negative infinity makes its final probability zero. For sorted probabilities, the reference's nucleus mask is:

$$
\text{mask}_i=\left(\sum_{j\le i}p_j-p_i>\text{top\_p}\right)
$$

The final softmax renormalizes the remaining mass. Top-p probabilities are computed after top-k, so the nucleus threshold applies to that already-filtered distribution.

## Pitfalls and reference scope

Scatter back before sampling or the returned index would refer to sorted order. Ties at the kth logit can retain more than `k` tokens. Because the top-p comparison is strict, a token whose preceding mass exactly equals the threshold is also retained. Small or nonpositive temperatures are clamped, rather than interpreted as a separate greedy mode. The reference does not validate `top_k <= V`, probability ranges, empty vocabularies, or invalid logits. Sampling is random; reproducible checks require an explicit PyTorch seed.
