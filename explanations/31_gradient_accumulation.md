# Gradient Accumulation

To train on a batch that does not fit in memory, split it into micro-batches and add their gradients before updating parameters. PyTorch already adds successive backward results into each parameter's `.grad`; this function controls when gradients are cleared, scaled, and used.

## Inputs and output

`model` is a trainable module, `optimizer` updates its parameters, and `loss_fn` returns a scalar tensor. `micro_batches` is a list of `(x, y)` pairs. Shapes depend on the model: our example uses input and target shapes `(1, 1)`. The function performs one optimizer update and returns a Python float containing the average micro-batch loss.

## Reading the actual code

```python
optimizer.zero_grad()
total_loss = 0.0
n = len(micro_batches)
```

Clearing gradients once at the beginning removes contributions from the previous training step. `n` is the number of micro-batches, not their total number of examples. `total_loss` is a reporting number; it does not participate in differentiation.

```python
for x, y in micro_batches:
    loss = loss_fn(model(x), y) / n
    loss.backward()
    total_loss += loss.item()
```

For each pair, `model(x)` produces predictions using the same parameter values. Dividing the scalar loss by `n` **before** backward also divides its gradients by `n`. The next backward pass adds another contribution without clearing the previous one. Calling `.item()` extracts the scaled loss as a Python number for reporting; it does not erase the gradients that backward has already computed.

Consider a bias-free model `prediction = w*x` starting at `w=0`. Two micro-batches contain `(x=1, y=2)` and `(x=3, y=6)`, and the loss is mean squared error. A single example has derivative `2*x*(w*x-y)`.

| Micro-batch | Prediction | Unscaled MSE | Loss divided by 2 | Gradient added to `w.grad` | Accumulated gradient |
| --- | ---: | ---: | ---: | ---: | ---: |
| `x=1, y=2` | 0 | 4 | 2 | -2 | -2 |
| `x=3, y=6` | 0 | 36 | 18 | -18 | -20 |

The full two-example mean loss is `(4+36)/2 = 20`. Its derivative at zero is `(-4-36)/2 = -20`, exactly the accumulated gradient. Omitting the division would give `-40`, doubling an SGD update at the same learning rate.

```python
optimizer.step()
return total_loss
```

The optimizer runs once after every backward pass. With SGD and `lr=0.1`, `w` becomes `0 - 0.1*(-20) = 2`. The returned loss is `2+18 = 20`, measured before that update. If we stepped inside the loop, the second forward would see a different `w`, and Adam would also advance its memories and step counter multiple times.

## Run the example

Load the reference solution first, then run:

```python
# Example: Match two accumulated micro-batches to one full batch
model = nn.Linear(1, 1, bias=False)
with torch.no_grad():
    model.weight.zero_()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
micro_batches = [(torch.tensor([[1.0]]), torch.tensor([[2.0]])),
                 (torch.tensor([[3.0]]), torch.tensor([[6.0]]))]
reported_loss = accumulated_step(model, optimizer, nn.MSELoss(), micro_batches)
assert reported_loss == 20.0
torch.testing.assert_close(model.weight.grad, torch.tensor([[-20.0]]))
torch.testing.assert_close(model.weight, torch.tensor([[2.0]]))
full_x = torch.tensor([[1.0], [3.0]])
full_y = torch.tensor([[2.0], [6.0]])
torch.testing.assert_close(model(full_x), full_y)
```

The final equality happens because this tiny example lies exactly on the fitted line after one particular SGD step. Gradient accumulation itself does not guarantee one-step convergence.

## Common mistakes and reference limits

Clearing gradients inside the loop discards earlier contributions. Equal weighting of micro-batch means equals a full-batch mean when micro-batches have equal sizes; unequal sizes require weighting by example count. BatchNorm and other batch-dependent behavior can also break full-batch equivalence. The reference expects a scalar loss, does not set training mode, and has no nonempty-list guard. With no micro-batches, it still clears gradients, calls `step()`, and returns `0.0`.
