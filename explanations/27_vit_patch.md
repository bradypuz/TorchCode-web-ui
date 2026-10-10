# Vision Transformer patch embedding

## What the layer does

Turn an image into a sequence of small image patches, then turn each patch into a learned feature vector. A Vision Transformer can process those vectors much as a language transformer processes token vectors.

The input image `x` has shape `(B, C, H, W)`: batch size, channels, height, and width. A patch has side length `p = patch_size`. If both image dimensions are divisible by `p`, there are `n_h = H // p` patch rows and `n_w = W // p` patch columns. The output shape is `(B, n_h * n_w, embed_dim)`. Each patch initially contains `C * p * p` numbers; `embed_dim` is the learned output vector's width.

## Prepare one projection shared by all patches

```python
self.patch_size = patch_size
self.num_patches = (img_size // patch_size) ** 2
self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
```

The constructor calls `super().__init__()` to register its linear layer. `num_patches` describes the configured square image. `proj` has weight shape `(embed_dim, C * p * p)` and a default bias. The same weights and bias are applied to every patch, allowing the layer to recognize similar local patterns at different locations.

This projection does not average by default. Each output feature can learn a different weighted combination of patch pixels, including contrasts, edges, or channel combinations.

## Split the spatial axes into patch and local coordinates

```python
B, C, H, W = x.shape
p = self.patch_size
n_h, n_w = H // p, W // p
x = x.reshape(B, C, n_h, p, n_w, p)
```

The first three lines read the actual image dimensions and count complete patches. The reshape replaces height by `(patch_row, local_row)` and width by `(patch_column, local_column)`. It changes the organization of axes, not pixel values.

```text
Original axes:  batch, channel, image_row, image_column
After reshape: batch, channel, patch_row, local_row, patch_column, local_column
Target axes:   batch, patch_row, patch_column, channel, local_row, local_column
```

`H` and `W` must be exact multiples of `p` for this reshape. The code does not crop a partial patch or add padding.

## Move whole-patch coordinates before pixel coordinates

```python
x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, n_h * n_w, C * p * p)
return self.proj(x)
```

The permutation performs the move shown above. The next reshape can then join patch row and patch column into the sequence axis, and join channel, local row, and local column into the feature axis.

Patch tokens are ordered left to right across the first patch row, then left to right across the next. Within a token, pixels are ordered channel first, then local row, then local column. The permutation is essential: simply flattening the first reshaped tensor would combine the wrong pixels into each patch.

Finally, `self.proj(x)` applies a linear transformation to the last dimension. It preserves batch and patch ordering while changing each patch vector from pixel count to embedding width.

## Trace a four-patch image

For a one-channel `4 × 4` image containing `1` through `16` in row order, use `2 × 2` patches. Set the first projection feature to the pixel sum and the second to `top_left - bottom_right + 1`.

| Token | Image region | Flattened pixel vector | Projected vector |
|---|---|---|---|
| `0` | top left | `[1, 2, 5, 6]` | `[14, -4]` |
| `1` | top right | `[3, 4, 7, 8]` | `[22, -4]` |
| `2` | bottom left | `[9, 10, 13, 14]` | `[46, -4]` |
| `3` | bottom right | `[11, 12, 15, 16]` | `[54, -4]` |

```python
# Example: Extract four local patches before projecting their sums and contrasts
layer = PatchEmbedding(img_size=4, patch_size=2, in_channels=1, embed_dim=2)
with torch.no_grad():
    layer.proj.weight.copy_(torch.tensor([[1., 1., 1., 1.], [1., 0., 0., -1.]]))
    layer.proj.bias.copy_(torch.tensor([0., 1.]))
x = torch.arange(1., 17.).reshape(1, 1, 4, 4)
pixels = x.reshape(1, 1, 2, 2, 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(1, 4, 4)
expected_pixels = torch.tensor([[[1., 2., 5., 6.], [3., 4., 7., 8.],
                                 [9., 10., 13., 14.], [11., 12., 15., 16.]]])
assert torch.equal(pixels, expected_pixels)
out = layer(x)
assert out.shape == (1, 4, 2)
assert torch.equal(out, torch.tensor([[[14., -4.], [22., -4.], [46., -4.], [54., -4.]]]))
```

For a second example, one `2 × 2` patch with two channels contains eight numbers. An identity projection exposes their exact order: all four pixels of channel zero precede all four pixels of channel one.

```python
# Example: Preserve channel-first pixel ordering inside one patch
layer = PatchEmbedding(img_size=2, patch_size=2, in_channels=2, embed_dim=8)
with torch.no_grad():
    layer.proj.weight.copy_(torch.eye(8))
    layer.proj.bias.zero_()
x = torch.tensor([[[[1., 2.], [3., 4.]], [[10., 20.], [30., 40.]]]])
out = layer(x)
assert out.shape == (1, 1, 8)
assert torch.equal(out, torch.tensor([[[1., 2., 3., 4., 10., 20., 30., 40.]]]))
```

## Common mistakes and reference scope

Preserve both the token order and the channel/local-pixel flattening order. This reference adds neither a class token nor positional embeddings. Forward uses the actual `H` and `W`, while `num_patches` is only stored metadata based on `img_size`; inconsistent image sizes can therefore make that metadata misleading even when the reshape succeeds.
