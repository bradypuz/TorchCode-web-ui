# Adam Optimizer

## Intuition

Adam keeps two memories for every parameter: a smoothed gradient that gives a direction, and a smoothed squared gradient that sets the update's scale. Parameters with consistently large gradients get a larger denominator; the first moment also smooths noisy changes in direction.

## Inputs and outputs

`MyAdam` receives an iterable of parameter tensors of arbitrary shapes. Each parameter's `grad`, first moment `m`, and second moment `v` have that parameter's shape. `step()` updates parameters in place and returns nothing. `zero_grad()` resets existing gradients to zero.

## Implementation walkthrough

1. Materialize `params` as a list so it can be visited repeatedly. Initialize `m` and `v` with `zeros_like` to preserve each parameter's shape, dtype, and device.
2. Increment the shared step counter `t` once per optimizer step, then enter `torch.no_grad()` so updating parameters does not create an autograd graph.
3. Skip parameters whose gradient is `None`. For every other parameter, update both exponential moving averages from its current gradient.
4. Divide the moments by their bias-correction factors. Both memories started at zero, so their early estimates otherwise underestimate the moments.
5. Subtract the learning-rate-scaled, normalized first moment from the parameter. Keep epsilon outside the square root, exactly as in the reference.
6. Clear gradients when beginning the next training iteration; PyTorch backward calls accumulate into existing gradients.

## Why it works

For gradient $g_t$, the update is:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad\hat v_t=\frac{v_t}{1-\beta_2^t},\qquad p_t=p_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

The judge compares repeated updates against `torch.optim.Adam` with matching hyperparameters and checks gradient clearing.

## Pitfalls and reference scope

Reuse the same optimizer across iterations: recreating it discards the moments and step counter. `step()` does not clear gradients automatically. This implementation has one global counter, even when some parameters lack gradients; PyTorch Adam tracks steps per parameter, so intermittent gradients can differ. The reference does not implement weight decay, parameter groups, sparse-gradient support, or hyperparameter validation.
