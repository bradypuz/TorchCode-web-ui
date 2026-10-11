This function implements **RMS normalization (RMSNorm)**: it rescales a tensor using its root mean square, then applies a per-feature scaling factor.

```python
def rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * weight
```

## 1. Calculate the root mean square

```python
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
```

Read this from the inside out:

- **`x.pow(2)`** squares every element.
- **`.mean(dim=-1, keepdim=True)`** averages those squares along the **last dimension**.
  - `dim=-1` means “the last axis.”
  - `keepdim=True` preserves that axis with size `1`, so the result can broadcast correctly when dividing `x`.
- **`+ eps`** adds a small positive number for numerical stability, preventing division by zero when the input is all zeros.
- **`torch.sqrt(...)`** takes the square root.

For a vector with \(d\) elements, the denominator is:

\[
\mathrm{rms} = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}
\]

For example, if `x` has shape `[batch, sequence_length, hidden_size]`, `rms` has shape `[batch, sequence_length, 1]`. Each token’s hidden-state vector gets its **own** normalization factor.

## 2. Normalize and scale

```python
return x / rms * weight
```

- **`x / rms`** rescales each vector so its root mean square is approximately `1`—the small `eps` makes this approximate.
- **`* weight`** applies a scaling factor, usually a vector of shape `[hidden_size]`, with one value per feature.

In a neural network, `weight` is typically a learnable parameter. This function itself does not create that parameter; it uses whatever tensor you pass in.

The complete formula is:

\[
y_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2 + \epsilon}}\,w_i
\]

## Small example

With `x = [3, 4]` and `weight = [1, 1]`, ignoring the tiny `eps`:

```text
Squared values:   [9, 16]
Mean square:      (9 + 16) / 2 = 12.5
Root mean square: sqrt(12.5) ≈ 3.536
Output:           [3 / 3.536, 4 / 3.536] ≈ [0.849, 1.131]
```

**Unlike LayerNorm, RMSNorm does not subtract the mean.** It adjusts the input’s magnitude without centering it around zero.