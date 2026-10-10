# Implement Dropout

## Goal and intuition

During training, randomly remove individual activations so the model cannot always rely on the same features. The surviving activations are enlarged to preserve their expected value. During evaluation, the layer returns its input unchanged. This convention is called **inverted dropout** because the scaling happens while training.

## Inputs and outputs

- `x`: a floating tensor of any shape.
- `p`: the probability of dropping each element; the reference's normal operating range is `0 <= p < 1`.
- Output: a tensor with the same shape as `x`.

The module has no learned weights. `nn.Module` supplies its `training` flag; `.train()` and `.eval()` switch the behavior. A newly constructed module starts in training mode.

## Walk through the reference

1. Store `p` in the constructor after initializing `nn.Module`.
2. If the module is evaluating, or if `p == 0`, return `x` immediately. This also avoids drawing unnecessary random numbers.
3. Generate one uniform random number for each element with `torch.rand_like(x)`.
4. Keep elements whose random draw is greater than `p`, and convert the Boolean mask to floating point.
5. Multiply by the mask and divide by `1 - p`. With `p=0.5`, surviving ones become twos, while dropped ones become zero.

## Why it works

For a fixed input element and a Bernoulli keep mask $m$:

$$
y=\frac{mx}{1-p},\qquad \mathbb{P}(m=1)=1-p
$$

$$
\mathbb{E}[y]=\frac{(1-p)x}{1-p}=x
$$

For a particular sampled mask, gradients are zero at dropped positions and scaled by `1/(1-p)` at kept positions. The expected-value argument does not mean every individual output has the same mean as its input.

## Pitfalls and reference scope

Do not apply dropout in evaluation mode, or divide by `p` instead of the keep probability. Masks change between calls unless randomness is controlled. This is elementwise dropout, not a shared channel or token mask. The reference **does not validate `p`** and **does not handle `p == 1`**: training at `p=1` divides zero by zero and produces `NaN`. Its `.float()` mask is float32 and can promote lower-precision inputs. The local judge covers evaluation identity, survivor scaling, approximate drop rate, and gradient flow, but not these boundaries.
