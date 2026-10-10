# Embedding Layer

## Goal and intuition

Replace discrete IDs with trainable vectors. An embedding is a table: every row represents one item, and every column is one coordinate of that item's representation. Looking up token ID 7 means retrieving row 7, not calculating a numerical function of the integer 7.

## Inputs and outputs

- `num_embeddings = N`: number of table rows.
- `embedding_dim = E`: width of each embedding vector.
- `indices`: integer tensor with any index shape, commonly `(B, S)`.
- `weight`: trainable floating tensor `(N, E)`.
- Output: `indices.shape + (E,)`; a `(B, S)` index tensor becomes `(B, S, E)`.

The integer IDs are selectors. Gradients train the selected rows of `weight`; integer indices themselves do not receive gradients.

## Walk through the reference

1. Initialize the module through `super().__init__()`.
2. Draw a random normal matrix of shape `(num_embeddings, embedding_dim)` using `torch.randn`.
3. Wrap that matrix in `nn.Parameter`. This registers it with the module and makes it trainable and visible to an optimizer.
4. In `forward`, return `self.weight[indices]`. PyTorch's tensor indexing gathers all requested rows while preserving the arrangement of the index tensor.

## Why it works

For an index $i$, the operation simply returns its table row:

$$
\operatorname{embedding}(i)=W_{i,:}
$$

This is equivalent to multiplying a one-hot vector by the weight matrix, but indexing avoids constructing a vocabulary-sized one-hot tensor. Backpropagation accumulates contributions into used rows. If an ID appears several times, its row receives the sum of those contributions. Unused rows receive zero gradient for that forward pass.

## Pitfalls and reference scope

A plain tensor attribute would not register the table as a model parameter; `nn.Parameter` is essential. Use integer index tensors of supported indexing types and keep indices on a compatible device. Valid vocabulary IDs are normally `0` through `N-1`. Because the implementation uses raw indexing, **negative indices wrap around** to rows at the end of the table rather than being rejected as vocabulary IDs. Positive out-of-range indices raise an indexing error. There is no padding-row behavior, sparse-gradient option, maximum-norm renormalization, or frequency-based gradient scaling. The judge checks table dimensions, batched lookups, and gradients for used and unused rows.
