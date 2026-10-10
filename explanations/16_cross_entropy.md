# Cross-Entropy Loss

## Goal and intuition

Turn a batch of class scores into a loss that penalizes low probability assigned to the correct class. The model supplies **logits**, not probabilities. Cross-entropy takes the negative logarithm of the correct class probability, then averages across examples. The reference computes log probabilities directly so it does not need to build a softmax and then take its logarithm.

## Inputs and outputs

- `logits`: floating tensor `(B, C)`, one row per example and one column per class.
- `targets`: integer tensor `(B,)`, containing the correct column index for each row.
- Return value: a scalar tensor containing the mean loss.

For example, targets `[2, 0]` select class 2 from row 0 and class 0 from row 1. They are class indices, not one-hot vectors.

## Walk through the reference

1. Compute `torch.logsumexp(logits, dim=-1, keepdim=True)`. The result has shape `(B, 1)` and is the log normalizer for each row.
2. Subtract that result from every logit. Broadcasting produces `(B, C)` log probabilities.
3. Build row indices with `torch.arange(targets.shape[0])`, and use paired row/target indexing to gather one log probability per example.
4. Average the selected values and negate the result.

## Why it works

$$
\log p_{b,c}=z_{b,c}-\log\sum_j e^{z_{b,j}}
$$

$$
L=-\frac{1}{B}\sum_{b=1}^{B}\log p_{b,y_b}
$$

`logsumexp` evaluates the logarithm of the exponential sum stably, using a shift rather than naively exponentiating large logits. Thus a confidently correct row such as `[1000, 0, 0]` can produce a finite loss near zero. Adding the same constant to every logit in a row does not change its probabilities or loss.

## Pitfalls and reference scope

Keep the normalizer dimension so subtraction broadcasts across classes. Passing already normalized probabilities would apply the wrong transformation. The reference supports this particular `(B, C)` integer-target task: it does not implement class weights, label smoothing, ignored labels, soft targets, or selectable reductions. It does not validate target bounds or empty batches; an empty mean is undefined and produces `NaN`. The judge checks agreement with the ordinary PyTorch loss, large-logit stability, scalar output, and gradients.
