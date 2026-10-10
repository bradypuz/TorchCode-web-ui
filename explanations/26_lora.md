# LoRA (Low-Rank Adaptation)

## Goal and intuition

Adapt a frozen linear layer by learning a small low-rank correction. Instead of updating every entry of a full weight matrix, LoRA represents its change as the product of two narrower matrices. The reference keeps the base layer fixed and trains only these added parameters.

## Inputs and outputs

- Input `x`: `(..., in_features)`.
- Base weight: `(out_features, in_features)`; base bias: `(out_features,)`.
- `lora_A`: `(rank, in_features)`.
- `lora_B`: `(out_features, rank)`.
- Output: `(..., out_features)`.

The adapter has `rank * (in_features + out_features)` trainable entries. A small positive rank can make this much less than a full weight update.

## Walk through the reference

1. Create `self.linear` as a normal `nn.Linear` layer, then set both its weight and bias to `requires_grad=False`.
2. Initialize `lora_A` with independent normal samples scaled by **0.01**. Initialize `lora_B` to all zeros. Both are registered parameters.
3. Store `scaling = alpha / rank`.
4. Compute the base result `self.linear(x)`.
5. Multiply `x @ lora_A.T` to reduce its feature width to `rank`, then multiply by `lora_B.T` to expand it to `out_features`.
6. Scale that correction and add it to the base result.

## Why it works

PyTorch stores inputs as row vectors, so the implementation uses transposed matrices:

$$
y=xW_0^\top+b_0+\frac{\alpha}{r}xA^\top B^\top
$$

The effective weight change is $(\alpha/r)BA$, whose rank is at most $r$. Zero-initializing `B` makes the initial adapter contribution exactly zero, so the first output equals the base layer. Gradients still reach the adapter parameters; `A` initially has zero gradient because `B` is zero, while `B` can begin learning immediately.

## Pitfalls and reference scope

Both base **weight and bias** are frozen, but that does not prevent gradients from flowing to the input. Matrix ordering and transposes matter: `A` first reduces width and `B` restores it. The reference initializes a new random base layer rather than loading pretrained weights; it demonstrates the adaptation structure, not a complete fine-tuning workflow. It has no adapter dropout, weight merging, checkpoint management, or rank validation; `rank=0` causes division by zero. The judge checks frozen parameters, adapter shapes, zero-initialized behavior, gradients, and the explicit forward formula.
