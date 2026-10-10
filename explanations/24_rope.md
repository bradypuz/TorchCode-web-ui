# Rotary positional embeddings

## What this function does

Make each query and key depend on its position by rotating pairs of features. A rotation changes the direction of a two-number vector without changing its length. Position zero leaves every pair unchanged; later positions rotate pairs by progressively larger angles.

`q` and `k` both have shape `(B, S, D)`, where `B` is the batch size, `S` the sequence length, and `D` an even feature width. The function returns two tensors of those same shapes. The pairs are adjacent features: `(0, 1)`, `(2, 3)`, and so on. They are not the first half of the vector paired with the second half.

## Assign one angle to each position and feature pair

```python
B, S, D = q.shape
pos = torch.arange(S, device=q.device).unsqueeze(1).float()
dim = torch.arange(0, D, 2, device=q.device).float()
freqs = 1.0 / (10000.0 ** (dim / D))
angles = pos * freqs
cos_a = torch.cos(angles)
sin_a = torch.sin(angles)
```

`torch.arange(S)` lists positions `0` through `S - 1`. Adding an axis with `.unsqueeze(1)` makes `pos` have shape `(S, 1)`. `dim` lists the first feature index of each pair, so it has `D / 2` entries. For `D = 4`, it is `[0, 2]`.

The frequency formula gives `[1, 0.01]` when `D = 4`: the first pair rotates at one radian per position, and the second at one hundredth of a radian per position. One radian is about 57.3 degrees. Different rates let feature pairs represent position at different scales.

Multiplying `(S, 1)` by `(D / 2,)` uses broadcasting to produce `(S, D / 2)`. Each row is one position; each column is one feature pair. Sine and cosine have that same shape.

| Position | Pair `(0, 1)` angle for `D = 4` | Pair `(2, 3)` angle |
|---|---|---|
| `0` | `0` | `0` |
| `1` | `1` | `0.01` |
| `2` | `2` | `0.02` |

Positions and angles are created on `q`'s device. `B` is unpacked but does not enter the angle calculation: every batch item uses the same position schedule.

## Rotate adjacent feature pairs

```python
x1, x2 = x[..., 0::2], x[..., 1::2]
return torch.stack([x1 * cos_a - x2 * sin_a,
                    x1 * sin_a + x2 * cos_a], dim=-1).flatten(-2)
```

The nested `rotate(x)` function receives either `q` or `k`. `x[..., 0::2]` selects features `0, 2, 4, ...`; `x[..., 1::2]` selects `1, 3, 5, ...`. Both have shape `(B, S, D / 2)`. Matching entries form the original pairs.

For pair `(a, b)` and angle `theta`, the two displayed expressions compute `(a*cos(theta) - b*sin(theta), a*sin(theta) + b*cos(theta))`. This is the ordinary two-dimensional rotation formula. The `(S, D / 2)` angle arrays broadcast across the leading batch dimension, applying the same angle to that pair at that position in every batch item.

`torch.stack(..., dim=-1)` puts the two rotated coordinates next to each other, giving `(B, S, D / 2, 2)`. `.flatten(-2)` merges the pair and coordinate axes to restore `(B, S, D)` with the original adjacent-pair order. Finally, `return rotate(q), rotate(k)` applies identical position rules to both tensors.

```text
[a0, b0, a1, b1]
   ↓ even/odd slices
x1 = [a0, a1]       x2 = [b0, b1]
   ↓ pairwise rotation
[[rotated_a0, rotated_b0], [rotated_a1, rotated_b1]]
   ↓ flatten the last two axes
[rotated_a0, rotated_b0, rotated_a1, rotated_b1]
```

## Two explicit rotations

For `D = 2`, the only frequency is one. At position one, rotating `[1, 0]` gives `[cos(1), sin(1)] ≈ [0.5403, 0.8415]`. Rotating `[0, 1]` gives `[-sin(1), cos(1)] ≈ [-0.8415, 0.5403]`. At position zero, both remain unchanged.

```torchcode-visual
rope
```

Use the visual to inspect how position changes the angle while preserving a pair's length.

```python
# Example: Rotate query and key pairs by one radian at position one
q = torch.tensor([[[1., 0.], [1., 0.]]])
k = torch.tensor([[[0., 1.], [0., 1.]]])
q_rot, k_rot = apply_rope(q, k)
c = math.cos(1.)
s = math.sin(1.)
assert torch.allclose(q_rot, torch.tensor([[[1., 0.], [c, s]]]), atol=1e-6)
assert torch.allclose(k_rot, torch.tensor([[[0., 1.], [-s, c]]]), atol=1e-6)
assert torch.allclose(q_rot.square().sum(-1), q.square().sum(-1), atol=1e-6)
assert torch.allclose(k_rot.square().sum(-1), k.square().sum(-1), atol=1e-6)
```

A second example checks the slower pair and batch broadcasting. At position one, `[1, 0, 1, 0]` becomes approximately `[0.5403, 0.8415, 0.99995, 0.01000]`.

```python
# Example: Use two rotation speeds and broadcast angles across batches
q = torch.tensor([[[1., 0., 1., 0.], [1., 0., 1., 0.]]]).repeat(2, 1, 1)
k = q.clone()
q_rot, k_rot = apply_rope(q, k)
expected = torch.tensor([math.cos(1.), math.sin(1.), math.cos(0.01), math.sin(0.01)])
assert q_rot.shape == (2, 2, 4)
assert torch.allclose(q_rot[0, 1], expected, atol=1e-6)
assert torch.equal(q_rot[0], q_rot[1])
assert torch.equal(q_rot, k_rot)
assert torch.equal(q_rot[:, 0], q[:, 0])
```

Rotating a query and key at the same position preserves their dot product. At different positions, their dot product can reflect their relative angle, which supplies position information to attention.

## Common mistakes and reference scope

Require an even `D` and compatible query/key shapes and devices. The implementation accepts 3-D tensors and starts positions at zero each call; cached decoding would need a position offset. Its angle tensors are float32, which can promote lower-precision inputs. Adjacent pairing is part of this reference and must be preserved when reproducing it.
