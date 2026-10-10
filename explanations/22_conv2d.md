# 2D Convolution

## Goal and intuition

Slide a bank of learned kernels across an image. At each spatial location, multiply a local image patch by a kernel and sum over input channels and kernel coordinates. The reference extracts all sliding patches with `unfold`, then expresses their weighted sums in one `einsum` operation.

## Inputs and outputs

- `x`: `(B, C_in, H, W)`.
- `weight`: `(C_out, C_in, kH, kW)`.
- Optional `bias`: `(C_out,)`.
- `stride` and `padding`: scalar integers in this reference.
- Output: `(B, C_out, H_out, W_out)`.

For symmetric padding `p` and stride `s`:

$$
H_{\mathrm{out}}=\left\lfloor\frac{H+2p-kH}{s}\right\rfloor+1,\qquad W_{\mathrm{out}}=\left\lfloor\frac{W+2p-kW}{s}\right\rfloor+1
$$

## Walk through the reference

1. If padding is positive, use `F.pad(x, [padding] * 4)` to add zeros on all four spatial sides.
2. Read the image and kernel dimensions. The subsequent `H` and `W` already include padding.
3. Call `unfold` on image height, then width. The patch view has shape `(B, C_in, H_out, W_out, kH, kW)`.
4. Evaluate `einsum('bihwjk,oijk->bohw', patches, weight)`. The repeated indices `i`, `j`, and `k` are summed; batch, output channel, and output location remain.
5. If supplied, reshape the bias to `(1, C_out, 1, 1)` and broadcast it over all examples and spatial positions.

## Why it works

Each output is the dot product between one flattened input patch and one flattened output kernel. Unfolding arranges the patches; `einsum` performs the multiply-and-sum without explicit Python loops. Its operations remain differentiable with respect to the input and weights. As in PyTorch's `conv2d`, kernels are applied without spatial flipping; this operation is technically cross-correlation.

## Pitfalls and reference scope

The channel counts must match, and the kernel must fit the padded image. Tuple strides or paddings, dilation, groups, and other padding modes are not implemented. The local `H_out` and `W_out` variables document the dimensions but are not used to construct the output; `unfold` determines it. Invalid dimensions and strides are left to underlying tensor operations to reject. The judge compares the reference numerically with `F.conv2d`, including scalar stride, zero padding, bias, and gradient flow.
