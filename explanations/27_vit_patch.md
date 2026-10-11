This class converts an image into a **sequence of patch embeddings**—a common first step in a **Vision Transformer (ViT)**.

The idea is:

```text
Image
  → split into non-overlapping patches
  → flatten each patch into a vector
  → project each vector into an embedding
```

For example, a `224 × 224` RGB image split into `16 × 16` patches becomes **196 patches**, each of which can be represented by a **768-dimensional embedding**.

## 1. Initialization

```python
class PatchEmbedding(nn.Module):
```

This defines a PyTorch module. Inheriting from `nn.Module` lets PyTorch track its trainable parameters and use it as part of a larger neural network.

```python
def __init__(self, img_size, patch_size, in_channels, embed_dim):
    super().__init__()
```

The arguments are:

| Argument | Meaning | Example |
|---|---|---|
| `img_size` | Expected side length of a square image | `224` |
| `patch_size` | Side length of each square patch | `16` |
| `in_channels` | Number of image channels | `3` for RGB |
| `embed_dim` | Length of each output embedding | `768` |

### Store the patch size

```python
self.patch_size = patch_size
```

This is used later to split the input image.

### Calculate the expected number of patches

```python
self.num_patches = (img_size // patch_size) ** 2
```

For a square image, the number of patches along each side is:

```text
224 // 16 = 14
```

So the total number is:

```text
14 × 14 = 196 patches
```

**Important:** `self.num_patches` is stored as metadata. It is not actually used in this `forward()` method; the method calculates the patch grid from the input’s actual height and width.

### Create the learned projection

```python
self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
```

A patch contains `patch_size × patch_size` values **per channel**, so its flattened length is:

```text
in_channels × patch_size × patch_size
```

For an RGB `16 × 16` patch:

```text
3 × 16 × 16 = 768 values
```

The linear layer maps this vector to `embed_dim` values:

```text
Flattened patch                          Patch embedding
[in_channels × patch_size × patch_size]  →  [embed_dim]
```

This is a **learned transformation**, not just a reshape. The same linear layer is applied independently to every patch.

---

## 2. Read the input dimensions

```python
def forward(self, x):
    B, C, H, W = x.shape
```

The input is expected to have shape:

```text
[B, C, H, W]
```

where:

- `B`: batch size
- `C`: number of channels
- `H`: image height
- `W`: image width

For example, a batch of eight RGB images might have shape:

```text
[8, 3, 224, 224]
```

```python
p = self.patch_size
n_h, n_w = H // p, W // p
```

Here:

- `n_h` is the number of patch rows.
- `n_w` is the number of patch columns.

With `p = 16`:

```text
n_h = 224 // 16 = 14
n_w = 224 // 16 = 14
```

---

## 3. Split the image dimensions into patches

```python
x = x.reshape(B, C, n_h, p, n_w, p)
```

This separates each spatial dimension into:

- Which patch the pixel belongs to.
- Where the pixel lies inside that patch.

The shape changes from:

```text
[B, C, H, W]
```

to:

```text
[B, C, patch_rows, patch_height, patch_cols, patch_width]
```

For the example:

```text
[8, 3, 224, 224]
    ↓
[8, 3, 14, 16, 14, 16]
```

Notice that the patch-grid dimensions and the within-patch dimensions are currently interleaved.

---

## 4. Group and flatten each patch

```python
x = x.permute(0, 2, 4, 1, 3, 5)
```

`permute()` rearranges the dimensions:

```text
Before: [B, C, n_h, p, n_w, p]
After:  [B, n_h, n_w, C, p, p]
```

Now the dimensions describing the **patch grid** come first, and the dimensions describing **one patch’s contents** come last:

```text
[B, patch_rows, patch_cols, channels, patch_height, patch_width]
```

Then:

```python
.reshape(B, n_h * n_w, C * p * p)
```

This combines:

- `n_h × n_w` into the total number of patches.
- `C × p × p` into one flattened vector per patch.

The result is:

```text
[B, number_of_patches, flattened_patch_size]
```

For the example:

```text
[8, 14, 14, 3, 16, 16]
    ↓
[8, 196, 768]
```

Patches appear in **row-by-row order**: left to right across the first patch row, then the next row, and so on.

### A small visual example

For a single-channel `4 × 4` image with `2 × 2` patches:

```text
 a  b │ c  d
 e  f │ g  h
──────┼──────
 i  j │ k  l
 m  n │ o  p
```

The flattened patch sequence becomes:

```text
[a, b, e, f]
[c, d, g, h]
[i, j, m, n]
[k, l, o, p]
```

Each row is one patch vector.

---

## 5. Produce the embeddings

```python
return self.proj(x)
```

`nn.Linear` operates on the last dimension, so it maps each flattened patch to an embedding:

```text
[B, number_of_patches, C × p × p]
    ↓
[B, number_of_patches, embed_dim]
```

For each patch vector \(v\), it computes:

\[
e = Wv + b
\]

where \(W\) and \(b\) are trainable parameters shared across all patches.

Example usage:

```python
patch_embed = PatchEmbedding(
    img_size=224,
    patch_size=16,
    in_channels=3,
    embed_dim=512,
)

images = torch.randn(8, 3, 224, 224)
embeddings = patch_embed(images)

print(embeddings.shape)
# torch.Size([8, 196, 512])
```

## Things to keep in mind

- **Height and width must be divisible by `patch_size`.** Otherwise, the first `reshape()` fails; this code does not crop or pad the image.
- **The input channel count must match `in_channels`.** Otherwise, the linear layer receives the wrong vector size.
- Despite the square-image calculation in `__init__`, `forward()` can process **rectangular images** if both spatial dimensions are divisible by the patch size.
- This produces patch embeddings only. It does **not** add positional embeddings, a classification token, or attention layers.

**In short:** the output is a batch of patch-token sequences, with shape **`[batch_size, number_of_patches, embed_dim]`**, ready to be used by a Transformer.