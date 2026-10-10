# Gradient Accumulation

## Intuition

When a large batch does not fit in memory, process smaller micro-batches one at a time while keeping their gradients. PyTorch adds each backward pass into existing parameter gradients. A single optimizer update at the end uses the combined result.

## Inputs and output

`micro_batches` is a list of `(x, y)` pairs. Input and target shapes depend on the model and scalar `loss_fn`; for example, a linear model may take `(M, D)` inputs and `(M, O)` targets. The function changes model parameters through the supplied optimizer and returns the mean of the micro-batch losses as a Python `float`.

## Implementation walkthrough

1. Call `optimizer.zero_grad()` once before any forward pass. This removes gradients from the previous training step.
2. Store `n = len(micro_batches)` and initialize a numeric loss total.
3. For each pair, compute the model output and divide its scalar loss by `n` before calling `backward()`.
4. Add the scaled loss's `.item()` to the reporting total. Extracting this number does not change the already-computed gradient.
5. Call `optimizer.step()` exactly once after all micro-batches, then return the loss total.

## Why it works

For micro-batch losses $L_1,\ldots,L_n$, differentiation is linear:

$$
\nabla_\theta\left(\frac{1}{n}\sum_{i=1}^nL_i\right)=\sum_{i=1}^n\nabla_\theta\left(\frac{L_i}{n}\right)
$$

Scaling before backward makes the accumulated gradient an average instead of a sum. It also makes the returned sum of scaled losses the same micro-batch average. The judge compares the update against concatenating equal-sized micro-batches for a linear model with mean-squared loss.

## Pitfalls and reference scope

Clearing gradients inside the loop would discard earlier contributions. Stepping inside it would change parameters between micro-batches and would advance Adam's state multiple times, producing a different update. Conversely, omitting division by `n` makes the gradient `n` times larger.

Equal weighting of micro-batch means matches an overall example mean when micro-batches have equal sizes. Unequal sizes require different weighting, which this reference does not provide. Batch-dependent layers such as BatchNorm can also prevent exact equivalence to a concatenated batch. The function expects a scalar loss and does not change training mode. It has no nonempty-list guard: an empty list returns `0.0` after clearing gradients and calling `step()`.
