# INT8 Quantized Linear Layer

## Intuition

Store a trained linear layer's weights as small integers, with a separate scale that reconstructs approximate floating-point values. Giving each output channel its own scale avoids letting one unusually large row determine precision for the entire matrix.

## Inputs and outputs

`weight` has shape `(O, I)` and optional `bias` has shape `(O,)`. `weight_int8` retains `(O, I)`, while `scale` has shape `(O, 1)` for broadcasting across each row. An input `x` of shape `(..., I)` produces `(..., O)`. Quantization happens once during construction; forward passes dequantize and perform a floating-point matrix multiplication.

## Implementation walkthrough

1. Find the largest absolute weight in each output row with `amax(dim=1, keepdim=True)`. Divide by `127.0` to obtain that row's scale.
2. Divide weights by `scale + 1e-10`, round to the nearest integer, clamp to `[-128, 127]`, and convert to `torch.int8`.
3. Register the integer weights and scale as buffers, so they appear in the module's state and follow device moves without appearing in `named_parameters()`.
4. If a bias is provided, clone it into an `nn.Parameter`.
5. In `forward`, convert integer weights to floating point and multiply by the stored scales. Multiply `x` by the transpose of the reconstructed weight matrix, then add the bias if present.

## Why it works

For output channel $o$:

$$
s_o=\frac{\max_i|W_{oi}|}{127},\qquad q_{oi}=\operatorname{clip}\left(\operatorname{round}\left(\frac{W_{oi}}{s_o+10^{-10}}\right),-128,127\right)
$$

$$
\widetilde W_{oi}=q_{oi}s_o,\qquad Y=X\widetilde W^\top+b
$$

Rounding introduces a small approximation error instead of preserving every original float. Keeping the channel axis allows different rows to use different integer step sizes. The judge checks INT8 storage, value bounds, reconstruction error, output shape, and buffer registration.

## Pitfalls and reference scope

Preserve `keepdim=True`; a bare `(O,)` scale would broadcast along the wrong axis. A zero row has zero scale and reconstructs as zeros; epsilon protects its quantization division. This is weight storage quantization with floating-point computation, not an integer-only kernel, and it does not quantize activations. Optional bias remains trainable. The reference does not explicitly detach or disable autograd during construction, so a scale computed from a weight requiring gradients can retain a link to that source tensor. Its `.float()` reconstruction also assumes compatible input dtype/device rather than providing general mixed-precision handling.
