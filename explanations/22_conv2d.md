# Two-dimensional convolution

## What this function computes

Slide each learned kernel across an image. At every position, multiply matching image and kernel entries, add those products across all input channels, and optionally add a bias. Each kernel produces one output channel.

`x` has shape `(B, C_in, H, W)`: batch size, input channels, height, and width. `weight` has shape `(C_out, C_in, kH, kW)`: output channels, input channels, and kernel height and width. Optional `bias` has shape `(C_out,)`. The output shape is `(B, C_out, H_out, W_out)`.

Here `stride` is an integer controlling how far the kernel moves each step, and `padding` is an integer counting zeros added to each edge. This operation follows PyTorch's usual convention: it does **not** reverse the kernel before multiplying.

## Pad first and find the output positions

```python
if padding > 0:
    x = F.pad(x, [padding] * 4)
B, C_in, H, W = x.shape
C_out, _, kH, kW = weight.shape
H_out = (H - kH) // stride + 1
W_out = (W - kW) // stride + 1
```

For an image, `F.pad` reads its four entries as left, right, top, and bottom. `[padding] * 4` therefore adds the same number of zero columns or rows on all sides. The subsequent `H` and `W` already include this padding.

A kernel of height `kH` can begin at rows `0, stride, 2 * stride, ...` while still fitting. Counting those positions gives the height formula. `//` is integer division, so incomplete final steps are discarded. The width formula follows the same rule. These two variables explain the expected output dimensions; the reference does not explicitly use them to reshape the result.

## Expose every patch with `unfold`

```python
patches = x.unfold(2, kH, stride).unfold(3, kW, stride)
```

The first `unfold` operates on image height, axis 2. It creates an axis for output row positions and appends an axis holding the kernel's local rows. The second operates on width, still axis 3, and appends the kernel's local columns. The final layout is:

```text
x:        (B, C_in, H,     W)
patches:  (B, C_in, H_out, W_out, kH, kW)
            b    i     h      w    j   k
```

Thus `patches[b, i, h, w, j, k]` is channel `i` at local row `j`, local column `k` inside the patch whose output position is `(h, w)`. `unfold` exposes overlapping windows; it does not average them or perform the multiplication yet.

## Multiply and sum the matching axes

```python
out = torch.einsum('bihwjk,oijk->bohw', patches, weight)
```

The labels describe the axes just shown. The weight's `o` is an output channel, and its `i`, `j`, `k` match the patch's input-channel and local-kernel axes. Labels absent from `bohw` are summed out. For one output number, this means summing `patches[b, i, h, w, j, k] * weight[o, i, j, k]` over every `i`, `j`, and `k`.

This is equivalent to matrix multiplication after arranging each patch as one row. Flatten in the order **input channel, kernel row, kernel column**, and flatten each kernel in that same order. Output positions come first; directly flattening the current six-axis tensor without moving its axes would mix the wrong positions.

## A patch-by-patch numerical example

For image `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]`, kernel `[[1, 2], [3, 4]]`, and stride one:

| Output position | Flattened patch | Dot product with `[1, 2, 3, 4]` | With bias `1` |
|---|---|---|---|
| `(0, 0)` | `[1, 2, 4, 5]` | `1 + 4 + 12 + 20 = 37` | `38` |
| `(0, 1)` | `[2, 3, 5, 6]` | `2 + 6 + 15 + 24 = 47` | `48` |
| `(1, 0)` | `[4, 5, 7, 8]` | `4 + 10 + 21 + 32 = 67` | `68` |
| `(1, 1)` | `[5, 6, 8, 9]` | `5 + 12 + 24 + 36 = 77` | `78` |

```python
# Example: Match unfolded patches to a matrix multiplication
x = torch.arange(1., 10.).reshape(1, 1, 3, 3)
weight = torch.tensor([[[[1., 2.], [3., 4.]]]])
bias = torch.tensor([1.])
out = my_conv2d(x, weight, bias)
assert torch.equal(out, torch.tensor([[[[38., 48.], [68., 78.]]]]))
patches = x.unfold(2, 2, 1).unfold(3, 2, 1)
assert patches.shape == (1, 1, 2, 2, 2, 2)
rows = patches.permute(0, 2, 3, 1, 4, 5).reshape(1, 4, 4)
assert torch.equal(rows[0], torch.tensor([[1., 2., 4., 5.], [2., 3., 5., 6.],
                                         [4., 5., 7., 8.], [5., 6., 8., 9.]]))
products = rows @ weight.reshape(1, 4).T + bias
assert torch.equal(products.transpose(1, 2).reshape(1, 1, 2, 2), out)
```

The bias is added by this final branch:

```python
if bias is not None:
    out = out + bias.view(1, -1, 1, 1)
```

The reshaped bias has one value per output channel and dimensions of size one for batch and space. Broadcasting repeats that channel's bias at every image position.

For a second example, pad a `2 × 2` image with one zero border and use stride two. Each `2 × 2` window then contains one original pixel and three zeros, so an all-ones kernel returns the original four values:

```python
# Example: Inspect border padding and a stride of two
x = torch.tensor([[[[1., 2.], [3., 4.]]]])
weight = torch.ones(1, 1, 2, 2)
out = my_conv2d(x, weight, stride=2, padding=1)
assert out.shape == (1, 1, 2, 2)
assert torch.equal(out, x)
```

## Common mistakes and reference scope

Keep the patch and kernel flattening orders identical, and sum over all input channels. The reference supports scalar stride and symmetric zero padding. It does not implement dilation, groups, separate height/width strides, or explicit validation of incompatible shapes.
