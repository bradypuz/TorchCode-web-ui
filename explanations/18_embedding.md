This code implements a **trainable embedding layer**: it takes integer IDs and looks up a vector for each ID.

For example, if words have been assigned IDs such as `"cat" → 0` and `"dog" → 1`, the layer stores one learnable vector for each word.

## 1. Define the layer

```python
class MyEmbedding(nn.Module):
```

`MyEmbedding` inherits from `nn.Module`, the base class for PyTorch neural-network layers.

This lets PyTorch manage its parameters and include it inside larger models.

## 2. Create the embedding table

```python
def __init__(self, num_embeddings, embedding_dim):
    super().__init__()
    self.weight = nn.Parameter(
        torch.randn(num_embeddings, embedding_dim)
    )
```

### The arguments

- **`num_embeddings`**: how many different IDs the layer can represent. For words, this might be the vocabulary size.
- **`embedding_dim`**: how many numbers are in each embedding vector.

For example:

```python
embedding = MyEmbedding(num_embeddings=5, embedding_dim=3)
```

creates a weight matrix with shape:

```text
(5, 3)
```

Think of it as a table:

```text
ID    Embedding vector
0     [a₀, a₁, a₂]
1     [b₀, b₁, b₂]
2     [c₀, c₁, c₂]
3     [d₀, d₁, d₂]
4     [e₀, e₁, e₂]
```

Each row belongs to one ID.

### What the individual operations do

- `super().__init__()` initializes the `nn.Module` machinery. This must happen before registering parameters.
- `torch.randn(...)` fills the table with random numbers drawn from a standard normal distribution.
- `nn.Parameter(...)` marks the table as a **trainable model parameter**. It will appear in `embedding.parameters()`, so an optimizer can update it.

Initially, these vectors are random; they acquire useful representations through training.

## 3. Look up the requested vectors

```python
def forward(self, indices):
    return self.weight[indices]
```

This selects rows from the weight matrix using the supplied IDs.

For example:

```python
embedding = MyEmbedding(num_embeddings=5, embedding_dim=3)

indices = torch.tensor([2, 0, 2], dtype=torch.long)
output = embedding(indices)

print(output.shape)  # torch.Size([3, 3])
```

Calling `embedding(indices)` invokes the module’s `forward` method through PyTorch’s module-call machinery.

The result contains:

```text
[
    the vector stored in row 2,
    the vector stored in row 0,
    the vector stored in row 2
]
```

Notice that ID `2` appears twice, so its vector appears twice. **The layer performs a lookup, not a matrix multiplication.**

## 4. How the output shape works

For integer index tensors, the output shape is the input shape with `embedding_dim` appended:

```text
Input shape             Output shape
(number_of_ids,)        (number_of_ids, embedding_dim)
(batch_size, seq_len)   (batch_size, seq_len, embedding_dim)
```

For example:

```python
embedding = MyEmbedding(num_embeddings=10, embedding_dim=4)

indices = torch.tensor([
    [1, 3, 1],
    [0, 5, 2],
])

output = embedding(indices)

print(output.shape)  # torch.Size([2, 3, 4])
```

That means **2 sequences × 3 IDs per sequence × 4 numbers per embedding**.

## 5. How it learns

When the output contributes to a loss and you call `loss.backward()`, PyTorch computes gradients for the selected rows of `self.weight`.

- Repeated IDs contribute gradients to the same row, and those contributions accumulate.
- Rows not selected receive no gradient contribution from this lookup.
- An optimizer’s `step()` then updates the parameters; calling `forward` alone does not train them.
- The integer IDs themselves are not learned.

**In short:** this is a minimal implementation of the basic lookup behavior of `nn.Embedding(num_embeddings, embedding_dim)`. The built-in version additionally supports features such as padding handling, norm constraints, and sparse gradients.