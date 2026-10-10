# Cosine Learning Rate Schedule with Warmup

## Intuition

A training run often starts with a cautious learning rate, ramps up to its peak, then gradually reduces the update size. This function joins a linear warmup to a cosine decay and holds the final learning rate after training ends.

## Inputs and output

`step`, `total_steps`, and `warmup_steps` are scalar counts. `max_lr` and `min_lr` are scalar learning rates. The output is a single numeric learning rate; there are no tensors or batch dimensions. The function only computes a value. The training loop must assign that value to the optimizer's learning-rate setting.

## Implementation walkthrough

1. Check whether `step < warmup_steps`. In this branch, return `max_lr * step / warmup_steps`, giving a straight ramp from zero.
2. Check whether `step >= total_steps`. Return `min_lr` here so the cosine does not begin rising again beyond the planned run.
3. For the remaining steps, measure progress over the decay interval rather than the whole training run: subtract warmup from both the current step and the interval length.
4. Map that progress through `cos(pi * progress)`, scale it to the learning-rate range, and add the minimum rate.

## Why it works

Let $s$ be the step, $W$ the warmup length, and $T$ the total length. With a valid schedule $0\le W<T$:

$$
\eta(s)=\begin{cases}
\eta_{\max}s/W & s<W\\
\eta_{\min} & s\ge T\\
\eta_{\min}+\frac{\eta_{\max}-\eta_{\min}}{2}\left(1+\cos\left(\pi\frac{s-W}{T-W}\right)\right) & \text{otherwise}
\end{cases}
$$

At the warmup boundary, decay progress is zero and the cosine branch returns `max_lr`. Halfway through decay it returns the midpoint of the two rates. At the end it reaches `min_lr`. The judge checks these boundaries, warmup monotonicity, and an interior cosine value.

## Pitfalls and reference scope

Use decay progress `(step - warmup_steps) / (total_steps - warmup_steps)`, not `step / total_steps`. Zero warmup works for nonnegative steps because its branch is skipped. The reference does not validate counts or rate ordering: negative steps and warmup extending past the total run can produce unintended values. In particular, the warmup branch is checked before the end-of-run branch. This is a stateless function, without optimizer integration or a stored step counter.
