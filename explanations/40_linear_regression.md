# Linear Regression in Three Ways

Fit predictions `X @ w + b` to observed targets by minimizing mean squared error. The reference contains **three** methods: a direct least-squares solve, manual gradient descent, and a PyTorch linear layer trained with autograd. They differ in how they find the same weights and bias.

## Inputs and outputs

`X` has shape `(N, D)`: examples by features. `y` has shape `(N,)`. Each method returns weight `w` of shape `(D,)` and scalar bias `b` of shape `()`. Predictions have shape `(N,)`. The iterative methods also accept learning rate `lr` and iteration count `steps`.

## Method one: solve least squares directly

```python
N, D = X.shape
X_aug = torch.cat([X, torch.ones(N, 1)], dim=1)
theta = torch.linalg.lstsq(X_aug, y).solution
w = theta[:D]
b = theta[D]
return w.detach(), b.detach()
```

Appending a ones column incorporates the bias into the weight vector. For one feature, `[x,1] @ [w,b]` equals `x*w+b`. `X_aug` has shape `(N,D+1)`, and `theta` has shape `(D+1,)`. `lstsq` finds the parameter vector with the smallest squared residual; despite the method's docstring mentioning the normal equation, the actual code does not explicitly invert `X.T @ X`. Slicing separates feature weights from the last bias entry, and detaching makes returned parameters independent of an autograd graph.

For `X=[[-1],[0],[1]]` and `y=[-1,1,3]`:

```text
X_aug = [[-1, 1],       theta = [2, 1]
         [ 0, 1],  ->   X_aug @ theta = [-1, 1, 3]
         [ 1, 1]]
```

## Method two: calculate gradients by hand

```python
w = torch.zeros(D)
b = torch.tensor(0.0)
for _ in range(steps):
    pred = X @ w + b
    error = pred - y
```

Zero weights and bias start every prediction at zero. `error` is the signed residual, not its square. For our three examples it is `[1,-1,-3]`.

```python
grad_w = (2.0 / N) * (X.T @ error)
grad_b = (2.0 / N) * error.sum()
w = w - lr * grad_w
b = b - lr * grad_b
```

The loss is the mean of squared residuals. Differentiating a square gives the factor two; taking a mean gives division by `N`. `X.T @ error` sums each feature's residual contribution and has shape `(D,)`. Bias affects every prediction equally, so its derivative sums all residuals.

| First manual step | Calculation | Result |
| --- | --- | ---: |
| Weight gradient | `(2/3)*((-1)*1 + 0*(-1) + 1*(-3))` | `-8/3` |
| Bias gradient | `(2/3)*(1-1-3)` | `-2` |
| Weight after `lr=0.1` | `0 - 0.1*(-8/3)` | `4/15` |
| Bias after `lr=0.1` | `0 - 0.1*(-2)` | `0.2` |

Repeated steps move toward `w=2,b=1`. This method calls neither backward nor an optimizer.

## Method three: let autograd calculate gradients

```python
layer = nn.Linear(D, 1)
optimizer = torch.optim.SGD(layer.parameters(), lr=lr)
loss_fn = nn.MSELoss()
```

The layer supplies trainable weights and a bias. Its initialization is random, unlike the manual method's zeros. SGD applies the same gradient-descent update rule; `MSELoss` uses a mean by default.

```python
optimizer.zero_grad()
pred = layer(X).squeeze(-1)
loss = loss_fn(pred, y)
loss.backward()
optimizer.step()
```

These statements run inside the iteration loop. Clearing gradients avoids accumulating across steps. The layer outputs `(N,1)`; squeezing the final singleton axis makes predictions `(N,)`, matching `y`. Backward computes derivatives and `step` updates the layer. Finally, `.weight.data.squeeze(0)` produces `(D,)`, and `.bias.data.squeeze(0)` produces a scalar.

## Run all three methods

Load the reference solution first, then run:

```python
# Example: Recover a line and check the manual first step
X = torch.tensor([[-1.0], [0.0], [1.0]])
y = torch.tensor([-1.0, 1.0, 3.0])
regression = LinearRegression()
first_w, first_b = regression.gradient_descent(X, y, lr=0.1, steps=1)
torch.testing.assert_close(first_w, torch.tensor([4/15]))
torch.testing.assert_close(first_b, torch.tensor(0.2))
torch.manual_seed(0)
results = [regression.closed_form(X, y),
           regression.gradient_descent(X, y, lr=0.1, steps=300),
           regression.nn_linear(X, y, lr=0.1, steps=300)]
for w, b in results:
    assert w.shape == (1,) and b.shape == ()
    torch.testing.assert_close(w, torch.tensor([2.0]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(b, torch.tensor(1.0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(X @ w + b, y, atol=1e-5, rtol=1e-5)
```

## Common mistakes and reference limits

Mixing predictions `(N,1)` with targets `(N,)` can broadcast an unintended `(N,N)` residual. Iterative convergence depends on feature scale, learning rate, and step count. Helper tensors and the layer use default dtype/device settings. There is no regularization or convergence test. The final `.data` outputs are views, not independent clones.
