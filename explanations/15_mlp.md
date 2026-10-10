# SwiGLU MLP

## Goal and intuition

Build a feed-forward layer that uses one learned projection to control another. Unlike a simple activation between two linear layers, SwiGLU has two parallel branches: the **gate** changes how strongly each intermediate feature contributes, while the **up** branch supplies its content. The final projection brings the result back to the model's width.

## Inputs and outputs

- Input `x`: `(B, S, d_model)` for token sequences, or `(B, d_model)` for a batch of vectors.
- Both intermediate branches: `(..., d_ff)`.
- Output: the same leading dimensions as `x`, with final width `d_model`.

The layers operate on the last dimension, so the sequence positions are processed independently. There is no interaction between tokens inside this MLP.

## Walk through the reference

1. Create `gate_proj` and `up_proj`, each mapping `d_model` to `d_ff`. Create `down_proj` mapping `d_ff` back to `d_model`.
2. Evaluate `gate_proj(x)` and apply `F.silu`. SiLU multiplies its input by its sigmoid, producing a smooth activation.
3. Evaluate `up_proj(x)` separately. Multiply its output element by element with the activated gate. These branches need matching shapes.
4. Pass the product through `down_proj` and return the result.

## Why it works

$$
\operatorname{SiLU}(z)=z\,\sigma(z)
$$

$$
y=W_{\mathrm{down}}\bigl(\operatorname{SiLU}(W_{\mathrm{gate}}x+b_{\mathrm{gate}})\odot(W_{\mathrm{up}}x+b_{\mathrm{up}})\bigr)+b_{\mathrm{down}}
$$

The multiplication lets the two projections interact before the final linear combination. The gate is not a probability: SiLU can be negative and is not restricted to `[0, 1]`. All projections and operations remain differentiable, so gradients reach the input and every parameter.

## Pitfalls and reference scope

Use elementwise multiplication, not a matrix product, between the branches. Applying SiLU to both branches would change the requested formula. The reference uses `nn.Linear` with its default **biases enabled**; reproducing a bias-free architecture would give different parameters and behavior. It includes no residual connection, normalization, or dropout. Those belong to a surrounding transformer block, not this exercise. The local judge checks parameter dimensions, 2-D and 3-D outputs, the explicit formula, and gradient flow.
