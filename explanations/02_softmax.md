This function implements **softmax**, which converts a set of scores into nonnegative values that sum to **1** along a chosen dimension. It is often used to turn model outputs, called *logits*, into probabilities.

For values \(x_1,\ldots,x_n\), softmax is:

\[
\operatorname{softmax}(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}
\]

Your code uses a **numerically stable** version of this formula.

## Line by line

### 1. Define the function
```python
def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
```

- **`x`**: The input tensor, normally containing floating-point scores.
- **`dim`**: The dimension along which to apply softmax. `-1` means the **last dimension**.
- **`-> torch.Tensor`**: A type annotation indicating that the function returns a tensor. Its shape will be the same as `x`.

For a 2D tensor, `dim=-1` applies softmax independently to each row.

### 2. Find the maximum along that dimension
```python
x_max = x.max(dim=dim, keepdim=True).values
```

This finds the largest value in each group being normalized.

- `.values` selects the maximum values; `torch.max` also returns their indices.
- `keepdim=True` preserves the reduced dimension with size `1`, so the result can broadcast correctly when subtracting it from `x`.

For example:
```text
x = [[1, 2, 3],
     [4, 5, 6]]

x_max = [[3],
         [6]]
```

### 3. Subtract the maximum and exponentiate
```python
e_x = torch.exp(x - x_max)
```

For the first row:
```text
Original:       [ 1,  2, 3]
Subtract max:   [-2, -1, 0]
Exponentiate:   [0.1353, 0.3679, 1.0000]
```

**Why subtract the maximum?** Exponentials grow very quickly. Computing `exp(1000)`, for example, overflows in common floating-point types.

Subtracting the maximum makes every shifted value nonpositive, so every exponential is at most `1`. Crucially, **subtracting the same constant from every score does not change softmax**:

\[
\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}
=
\frac{e^{x_i}e^{-m}}{e^{-m}\sum_j e^{x_j}}
=
\frac{e^{x_i}}{\sum_j e^{x_j}}
\]

### 4. Normalize by the sum
```python
return e_x / e_x.sum(dim=dim, keepdim=True)
```

This divides each exponential by the sum of the exponentials in its group:

```text
Sum:    0.1353 + 0.3679 + 1.0000 ≈ 1.5032
Result: [0.0900, 0.2447, 0.6652]
```

The largest input gets the largest output, and the outputs sum to approximately `1` because of floating-point rounding.

## Example

```python
x = torch.tensor([1.0, 2.0, 3.0])
print(my_softmax(x))

# tensor([0.0900, 0.2447, 0.6652])
```

For a matrix:

```python
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])

print(my_softmax(x, dim=-1))

# tensor([[0.0900, 0.2447, 0.6652],
#         [0.0900, 0.2447, 0.6652]])
```

Both rows produce the same result because the second row is just the first row plus a constant.

**In practice, this implements the same operation as `torch.softmax(x, dim=dim)` for ordinary finite inputs.** PyTorch’s built-in version is generally preferred because it is optimized.