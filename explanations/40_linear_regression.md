This class fits a **linear regression model in three different ways**:

1. Solve the least-squares problem directly.
2. Use gradient descent with manually calculated gradients.
3. Use PyTorch’s `nn.Linear` and automatic differentiation.

All three aim to learn the same model:

\[
\hat y = Xw + b
\]

where:

| Quantity | Shape | Meaning |
|---|---|---|
| `X` | `(N, D)` | `N` examples, each with `D` input features |
| `y` | `(N,)` | One target value per example |
| `w` | `(D,)` | One learned weight per feature |
| `b` | scalar | Learned bias, or intercept |
| `pred` | `(N,)` | Predicted values for all examples |

For one example with two features, the prediction is:

\[
\hat y_i = w_1 x_{i1} + w_2 x_{i2} + b
\]

The goal is to minimize the **mean squared error**:

\[
L(w,b)=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2
\]

---

## 1. `closed_form`: solve least squares directly

```python
N, D = X.shape
```

This extracts the number of examples and features.

### Include the bias in the matrix

```python
X_aug = torch.cat([X, torch.ones(N, 1)], dim=1)
```

A column of ones is added to `X`:

\[
X_{\text{aug}} =
\begin{bmatrix}
x_{11} & \cdots & x_{1D} & 1\\
x_{21} & \cdots & x_{2D} & 1\\
\vdots & & \vdots & \vdots\\
x_{N1} & \cdots & x_{ND} & 1
\end{bmatrix}
\]

The weights and bias can now be combined into one vector:

\[
\theta =
\begin{bmatrix}
w_1\\
\vdots\\
w_D\\
b
\end{bmatrix}
\]

Then:

\[
X_{\text{aug}}\theta = Xw+b
\]

The final column contributes `1 * b` to every prediction.

### Find the best parameters

```python
theta = torch.linalg.lstsq(X_aug, y).solution
```

`lstsq` finds a solution that minimizes:

\[
\|X_{\text{aug}}\theta-y\|_2^2
\]

Minimizing this sum of squared errors is equivalent to minimizing mean squared error, because dividing by `N` does not change the minimizing parameters.

**Important distinction:** although the comments mention the normal equation,

\[
X_{\text{aug}}^\top X_{\text{aug}}\theta
= X_{\text{aug}}^\top y,
\]

the code does **not** explicitly build or invert \(X_{\text{aug}}^\top X_{\text{aug}}\). It calls a least-squares solver directly, which is generally preferable numerically.

### Separate weights and bias

```python
w = theta[:D]
b = theta[D]
```

The first `D` values are feature weights, and the last value is the bias.

```python
return w.detach(), b.detach()
```

`detach()` disconnects the returned tensors from any autograd graph. It does not change their numerical values.

**Key idea:** this method solves for the parameters without a training loop or learning rate.

---

## 2. `gradient_descent`: calculate gradients manually

### Start with zero parameters

```python
w = torch.zeros(D)
b = torch.tensor(0.0)
```

Initially, all predictions are zero:

\[
\hat y = X(0)+0=0
\]

Unlike in many neural networks, zero initialization is perfectly reasonable for ordinary linear regression.

### Make predictions and calculate errors

```python
pred = X @ w + b
error = pred - y
```

- `X @ w` performs matrix–vector multiplication.
- The scalar bias is broadcast to every example.
- `error[i]` is the prediction minus the target for example `i`.

### Calculate the gradients

```python
grad_w = (2.0 / N) * (X.T @ error)
grad_b = (2.0 / N) * error.sum()
```

These are the derivatives of the mean squared error:

\[
\nabla_w L = \frac{2}{N}X^\top(Xw+b-y)
\]

\[
\frac{\partial L}{\partial b}
= \frac{2}{N}\sum_{i=1}^{N}(\hat y_i-y_i)
\]

The weight gradient measures how each feature contributes to the errors. The bias gradient sums the errors because changing the bias shifts every prediction equally.

### Update the parameters

```python
w = w - lr * grad_w
b = b - lr * grad_b
```

Gradient descent moves parameters **opposite the gradient**, toward lower loss:

\[
w \leftarrow w-\text{lr}\,\nabla_w L
\]

\[
b \leftarrow b-\text{lr}\,\frac{\partial L}{\partial b}
\]

This repeats `steps` times.

- `lr` controls the size of each update.
- `steps` controls the number of updates.
- Too small a learning rate can make convergence slow.
- Too large a learning rate can cause oscillation or divergence.

**Key idea:** the gradients are written out explicitly; there is no call to `loss.backward()`.

Strictly speaking, “no autograd” here means **autograd is not used to calculate the gradients**. If `X` or `y` has `requires_grad=True`, PyTorch can still track these operations unless you use `torch.no_grad()`.

---

## 3. `nn_linear`: let PyTorch calculate the gradients

### Create the model

```python
layer = nn.Linear(D, 1)
```

This creates a linear layer with:

- `D` input features
- `1` output value per example
- A learnable weight tensor of shape `(1, D)`
- A learnable bias tensor of shape `(1,)`

It implements the same model as before:

\[
\hat y = Xw+b
\]

Unlike the manual method, `nn.Linear` initializes its parameters randomly by default.

### Create the optimizer and loss function

```python
optimizer = torch.optim.SGD(layer.parameters(), lr=lr)
loss_fn = nn.MSELoss()
```

- `SGD` updates the layer’s weights and bias using their gradients.
- `MSELoss()` computes mean squared error by default.

Despite the name “stochastic gradient descent,” this code uses **the entire dataset on every step**, so the training loop is full-batch gradient descent.

### Run the training loop

```python
optimizer.zero_grad()
```

Clear previously stored gradients. PyTorch normally **accumulates** gradients rather than replacing them on each backward pass.

```python
pred = layer(X).squeeze(-1)
```

`layer(X)` has shape `(N, 1)`. `squeeze(-1)` removes the final dimension of size one, producing `(N,)` to match `y`.

```python
loss = loss_fn(pred, y)
```

Compute the mean squared error.

```python
loss.backward()
```

Autograd calculates the gradients of the loss with respect to the layer’s weights and bias.

```python
optimizer.step()
```

The optimizer updates those parameters. With the default SGD settings used here, the update is the same basic rule as in the manual method:

\[
\text{parameter}
\leftarrow
\text{parameter}-\text{lr}\times\text{gradient}
\]

### Extract the learned parameters

```python
w = layer.weight.data.squeeze(0)
b = layer.bias.data.squeeze(0)
```

The shapes change as follows:

- Weight: `(1, D)` → `(D,)`
- Bias: `(1,)` → scalar

However, **`.data` is discouraged** because it bypasses autograd’s safety checks. Prefer:

```python
w = layer.weight.detach().squeeze(0)
b = layer.bias.detach().squeeze(0)
```

---

## How the methods compare

| Method | How it finds parameters | Uses autograd for gradients? | Main considerations |
|---|---|---|---|
| `closed_form` | Direct least-squares solve | No training loop | Solver cost and matrix rank |
| `gradient_descent` | Repeated manual updates | No | Learning rate and number of steps |
| `nn_linear` | Repeated optimizer updates | Yes | Learning rate, steps, initialization |

With a suitable learning rate and enough iterations, the iterative methods should approach a least-squares optimum.

They will not necessarily return exactly the same parameters:

- The iterative methods may not have converged yet.
- Their initializations differ.
- Numerical precision affects results.
- If the problem has multiple optimal parameter vectors, different methods can find different ones with equally good predictions.

If you give the manual method and `nn.Linear` the **same initial weights and bias**, their updates should closely match under the same learning rate and loss.

## Two practical fixes

### Match the input’s device and dtype

As written, the newly created tensors and layer use default settings, which can fail if `X` is on a GPU or uses a different dtype. Safer initialization is:

```python
# In closed_form:
X_aug = torch.cat([X, X.new_ones((N, 1))], dim=1)

# In gradient_descent:
w = X.new_zeros(D)
b = X.new_zeros(())

# In nn_linear:
layer = nn.Linear(D, 1).to(device=X.device, dtype=X.dtype)
```

`y` should also have a compatible device and dtype.

### Keep the target shape consistent

This code expects:

```python
X.shape == (N, D)
y.shape == (N,)
```

If `y` has shape `(N, 1)`, subtracting it from predictions of shape `(N,)` can accidentally broadcast to an `(N, N)` tensor. For a single target per example, convert it with:

```python
y = y.squeeze(-1)
```

**In short:** the first method asks a solver for the best fit, the second computes each optimization step yourself, and the third lets PyTorch calculate and apply those steps.