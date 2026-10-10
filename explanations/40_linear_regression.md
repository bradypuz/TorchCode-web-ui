# Linear Regression in Three Ways

## Intuition

Fit one weight per input feature plus an intercept so predictions match observed targets. The reference solves the same least-squares objective in three ways: direct linear algebra, hand-written gradient descent, and PyTorch's trainable linear layer.

## Inputs and outputs

`X` has shape `(N, D)` and `y` has shape `(N,)`. Each method returns `(w, b)` with weight shape `(D,)` and scalar bias shape `()`. Predictions are `X @ w + b`, shape `(N,)`. The iterative methods take a learning rate and number of steps.

## Implementation walkthrough

1. **Closed form:** append a column of ones to create `(N, D + 1)` augmented inputs. Call `torch.linalg.lstsq(X_aug, y).solution`, split its first `D` entries from the final bias, and detach both outputs. The implementation solves least squares directly rather than explicitly inverting a normal-equation matrix.
2. **Manual gradient descent:** start with zero weights and bias. Compute predictions and residuals. Calculate `grad_w = (2 / N) * X.T @ error` and `grad_b = (2 / N) * error.sum()`, then subtract the learning-rate-scaled gradients for each step. No backward call or optimizer is used.
3. **`nn.Linear`:** create `nn.Linear(D, 1)`, an SGD optimizer, and `nn.MSELoss`. Clear gradients, compute predictions with the last singleton dimension squeezed, backpropagate, and step. Extract weight and bias from the trained layer and squeeze to the required shapes.

## Why it works

All methods target:

$$
L(w,b)=\frac{1}{N}\|Xw+b\mathbf1-y\|_2^2
$$

$$
\nabla_wL=\frac{2}{N}X^\top(Xw+b\mathbf1-y),\qquad\frac{\partial L}{\partial b}=\frac{2}{N}\mathbf1^\top(Xw+b\mathbf1-y)
$$

Augmenting with ones folds the intercept into a single least-squares vector. Autograd obtains the same derivatives used in the manual method. The judge checks recovered parameters, shape correctness, agreement among the three approaches, and detached closed-form weights.

## Pitfalls and reference scope

Keep `y` and predictions one-dimensional; mixing `(N,)` with `(N, 1)` can broadcast into an unintended `(N, N)` error matrix. Iterative convergence depends on learning rate, feature scaling, and step count. The reference creates helper tensors and the layer using default device/dtype settings, so other devices or dtypes may require adaptation. It includes no regularization or convergence test. Its manual method avoids autograd APIs but does not explicitly detach inputs, and `nn_linear` returns `.data` views rather than independent clones.
