# ViT Patch Embedding

## Goal and intuition

Convert an image into a sequence of patch vectors that a transformer can process like tokens. Each non-overlapping square patch is flattened into a feature vector, then the same learned linear projection embeds every patch. This layer creates the patch tokens; it does not perform attention.

## Inputs and outputs

- Image `x`: `(B, C, H, W)`.
- Patch side length: `P = patch_size`.
- Patch count: `N = (H/P) * (W/P)` when both dimensions are divisible by `P`.
- Flattened patches: `(B, N, C * P * P)`.
- Output: `(B, N, embed_dim)`.

The constructor's `num_patches` attribute uses a square configured image: `(img_size // patch_size) ** 2`.

## Walk through the reference

1. Store `patch_size`, compute the configured patch count, and create `nn.Linear(in_channels * P * P, embed_dim)`.
2. In `forward`, calculate `n_h = H // P` and `n_w = W // P` from the actual image.
3. Reshape to `(B, C, n_h, P, n_w, P)`. This separates each spatial dimension into its patch-grid coordinate and its within-patch coordinate.
4. Permute to `(B, n_h, n_w, C, P, P)` so each patch's channel and pixel coordinates are grouped together.
5. Reshape to `(B, n_h * n_w, C * P * P)`, then apply the shared projection.

## Why it works

The reshape and permutation reorganize existing pixels without mixing their values. Afterward, the linear layer computes a learned weighted combination of the pixels within each patch:

$$
z_{b,n}=W_{\mathrm{proj}}\,\operatorname{vec}(\mathrm{patch}_{b,n})+b_{\mathrm{proj}}
$$

Patch tokens follow row-major grid order: left to right across each row, then top to bottom. Every patch uses the same projection, and gradients flow back through the reorganization to the image and projection parameters.

## Pitfalls and reference scope

Directly reshaping into `(B, N, C*P*P)` without the intermediate permutation would group pixels incorrectly. Actual height and width must be divisible by `P`; the reference does not crop or pad, and floor division alone does not make incompatible images valid. The actual channel count must match `in_channels`. Runtime rectangular images can work if divisible, but `num_patches` still reflects the constructor's square `img_size` and is not updated. There are no position embeddings, class token, normalization, or transformer blocks. Tests check output dimensions, the configured count, and gradient flow.
