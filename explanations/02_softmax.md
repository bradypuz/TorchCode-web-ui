# Implement Softmax

The function turns scores into proportions along a chosen axis. Every finite nonempty slice gets its own normalization: its returned values are non-negative and sum to one. The three statements do three jobs—find a safe offset, exponentiate, then divide by the total.

## Know which values belong to one distribution

`dim` is the axis whose entries compete with one another. It defaults to `-1`, meaning the last axis. For an input `(B, C)`, `B` is batch size and `C` is the number of classes. With `dim=-1`, each row is a separate distribution over classes. With `dim=0`, each column is normalized across examples instead. Both are valid tensor operations, but they answer different questions.

The output has the same shape as `x`. Reductions temporarily shrink the chosen axis to size one so their results can broadcast back into the original tensor.

## 1. Find one maximum per slice

```python
x_max = x.max(dim=dim, keepdim=True).values
```

`max` returns values and indices; `.values` selects just the largest scores. `keepdim=True` retains the reduced axis rather than removing it. On `(2, 3)` with `dim=-1`, the maxima have shape `(2, 1)`. Thus row zero's maximum is subtracted only from row zero, and row one's maximum only from row one.

## 2. Shift before exponentiating

```python
e_x = torch.exp(x - x_max)
```

Consider scores `[1000, 1000 + log(3)]`. Direct exponentiation would overflow. Subtracting the maximum gives `[-log(3), 0]`; exponentiating gives `[1/3, 1]`. Both are small, representable values.

Why may we subtract the maximum? Adding or subtracting a common constant changes all exponentials by the same factor, which cancels in the eventual division:

$$
\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}=\frac{e^{x_i}}{\sum_j e^{x_j}}.
$$

The offset changes numerical safety, not the mathematical distribution.

## 3. Divide by the matching sum

```python
return e_x / e_x.sum(dim=dim, keepdim=True)
```

The sum must use the **same axis** as the maximum. In the two-score example, the total is `4/3`, so dividing `[1/3, 1]` by it returns `[1/4, 3/4]`. `keepdim=True` again produces a denominator with a size-one axis. PyTorch broadcasts that one denominator to all entries within its slice.

This small matrix shows why the axis matters:

| Scores | Normalize each row, `dim=-1` | Normalize each column, `dim=0` |
|---|---|---|
| Row 0: `[0, 0]` | `[0.5, 0.5]` | `[0.5, 0.25]` |
| Row 1: `[0, log(3)]` | `[0.25, 0.75]` | `[0.5, 0.75]` |

The left output has rows summing to one. The right output has columns summing to one.

```python
# Example: The axis decides which entries compete
import math
x = torch.tensor([[0., 0.], [0., math.log(3.)]])
rows = my_softmax(x, dim=-1)
columns = my_softmax(x, dim=0)
assert torch.allclose(rows, torch.tensor([[0.5, 0.5], [0.25, 0.75]]))
assert torch.allclose(columns, torch.tensor([[0.5, 0.25], [0.5, 0.75]]))
assert torch.allclose(rows.sum(dim=-1), torch.ones(2))
assert torch.allclose(columns.sum(dim=0), torch.ones(2))
large = torch.tensor([1000., 1000. + math.log(3.)], dtype=torch.float64)
assert torch.allclose(my_softmax(large), torch.tensor([0.25, 0.75], dtype=torch.float64))
```

The example imports `math` just to construct `log(3)`; the actual softmax implementation uses only `torch`.

## Gradients and limits

Max, subtraction, exponentiation, sum, and division remain tensor operations, so the returned distribution stays connected to autograd. With finite nonempty slices, the maximum entry produces `exp(0)=1`, preventing a zero denominator caused solely by underflow. This does not handle every special value: positive infinity, an all-negative-infinity slice, or an empty reduction axis can still fail or produce `NaN`. The implementation also does not explicitly use a wider accumulation dtype.
