# Rotary Position Embedding (RoPE)

## Goal and intuition

Encode a token's position by rotating pairs of query and key coordinates. Different pairs rotate at different frequencies. Applying the same rotation family to queries and keys makes their dot product sensitive to their **relative position**, while preserving each vector's norm.

## Inputs and outputs

- `q` and `k`: matching tensors `(B, S, D)` on the same device.
- `D` must be even, so every coordinate has a partner.
- Return value: a pair `(q_rot, k_rot)` with the same shapes.
- Position angles and their sine/cosine tables: `(S, D/2)`, broadcast over the batch.

This reference uses consecutive coordinate pairs `(0, 1)`, `(2, 3)`, and so on.

## Walk through the reference

1. Build positions `0` through `S-1` as a column and even dimension indices `0, 2, ..., D-2` as a row.
2. Compute inverse frequencies `1 / 10000^(dim / D)`. Their outer product with positions gives one angle per position and pair.
3. Compute sine and cosine tables once, shared by queries and keys.
4. Inside `rotate`, slice even coordinates into `x1` and odd coordinates into `x2`.
5. Apply the two-dimensional rotation, stack the resulting paired coordinates, and flatten the last two dimensions to restore the original order.

## Why it works

For pair index $i$ and position $p$:

$$
\theta_{p,i}=p\,10000^{-2i/D}
$$

$$
\begin{pmatrix}x'_e\\x'_o\end{pmatrix}=\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}\begin{pmatrix}x_e\\x_o\end{pmatrix}
$$

Rotation matrices preserve length. In a query/key dot product, their composition depends on the difference between the two angles. Shifting both positions by the same offset therefore leaves that positional contribution unchanged; the actual vectors' content still matters.

## Pitfalls and reference scope

Keep each even coordinate adjacent to its odd partner when restoring the tensor. RoPE pairing conventions differ, so this consecutive-pair version must match the intended model. The code derives all tables from `q` and does not validate mismatched inputs or odd `D`. It starts positions at zero on every call, with no cache offset or supplied position IDs. Its float32 angle tables can promote half-precision inputs. It has no dedicated head axis, frequency cache, or long-context scaling. The judge checks shapes, norm preservation, a common-position shift, and gradients.
