This function computes **scaled dot-product attention** without building the entire attention matrix at once. It processes queries, keys, and values in small blocks and uses an **online softmax** to combine the results correctly.

That is the central idea behind FlashAttention—although this Python implementation is not the optimized, fused GPU kernel used by actual FlashAttention.

## 1. What it computes

Ordinary attention is:

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt D}\right)V
\]

Here, all three inputs are assumed to have shape:

```text
Q, K, V: [B, S, D]

B = batch size
S = sequence length
D = embedding / feature dimension
```

Conceptually:

- **Q — queries:** What each position is looking for.
- **K — keys:** What each position can be matched against.
- **V — values:** The information to combine after deciding which positions matter.

For each query, the function compares it with every key, turns those comparisons into softmax weights, and returns a weighted sum of the values.

The output has shape `[B, S, D]`.

---

## 2. Why process attention in blocks?

A straightforward implementation would look like:

```python
scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(D)
weights = torch.softmax(scores, dim=-1)
output = torch.bmm(weights, V)
```

The problem is that `scores` and `weights` each have shape `[B, S, S]`. Their memory usage grows **quadratically with sequence length**.

Your implementation instead computes small pieces of that matrix:

```python
for i in range(0, S, block_size):
    ...
    for j in range(0, S, block_size):
        ...
```

- The outer loop selects a **block of queries**.
- The inner loop visits **every block of keys and values**.

With `block_size=32`, each score block is usually only `[B, 32, 32]`, rather than `[B, S, S]`.

**Important:** Blocking does not remove any attention connections. Every query still attends to every key.

---

## 3. The outer loop: select queries and initialize running statistics

```python
B, S, D = Q.shape
output = torch.zeros_like(Q)
```

This reads the input dimensions and allocates the final result.

```python
for i in range(0, S, block_size):
    qi = Q[:, i:i+block_size]
    bs_q = qi.shape[1]
```

`qi` is the current query block, with shape:

```text
[B, bs_q, D]
```

`bs_q` may be smaller than `block_size` for the final block.

For each query in this block, the function initializes three running quantities:

```python
row_max = torch.full((B, bs_q, 1), float('-inf'), device=Q.device)
row_sum = torch.zeros(B, bs_q, 1, device=Q.device)
acc = torch.zeros(B, bs_q, D, device=Q.device)
```

| Variable | Meaning |
|---|---|
| `row_max` | Largest attention score seen so far for each query |
| `row_sum` | Running sum of exponentiated scores, used as the softmax denominator |
| `acc` | Running weighted sum of values, used as the numerator |

The trailing dimension of size `1` lets `row_max` and `row_sum` broadcast across keys or value features.

---

## 4. The inner loop: score one key block

```python
kj = K[:, j:j+block_size]
vj = V[:, j:j+block_size]
```

These select matching blocks of keys and values:

```text
kj: [B, bs_k, D]
vj: [B, bs_k, D]
```

Then:

```python
scores = torch.bmm(qi, kj.transpose(1, 2)) / math.sqrt(D)
```

`torch.bmm` performs a separate matrix multiplication for each batch item:

```text
[B, bs_q, D] × [B, D, bs_k]
                  ↓
           [B, bs_q, bs_k]
```

Each score is a query–key dot product divided by \(\sqrt D\). This scaling helps keep the scores from becoming excessively large as the feature dimension increases.

---

## 5. The key idea: online, numerically stable softmax

For one query, softmax normally computes:

\[
p_t=\frac{e^{s_t}}{\sum_u e^{s_u}}
\]

For numerical stability, we subtract the largest score \(m\):

\[
p_t=\frac{e^{s_t-m}}{\sum_u e^{s_u-m}}
\]

Subtracting the same number from every score does not change softmax, but it helps prevent exponential overflow.

The challenge here is that **the function has not seen all scores yet**. A later key block might contain a larger maximum.

### Find the updated maximum

```python
block_max = scores.max(dim=-1, keepdim=True).values
new_max = torch.maximum(row_max, block_max)
```

For each query:

- `block_max` is the largest score in the current key block.
- `new_max` is the largest score across all blocks processed so far.

### Rescale earlier contributions

```python
correction = torch.exp(row_max - new_max)
```

Previous contributions were calculated relative to the old maximum. They must now be expressed relative to the new maximum.

The reason this works is:

\[
e^{s-m_{\text{new}}}
=
e^{s-m_{\text{old}}}
\underbrace{e^{m_{\text{old}}-m_{\text{new}}}}_{\text{correction}}
\]

For example, if the running maximum increases from `5` to `7`, all previous exponential contributions are multiplied by \(e^{-2}\).

### Add the current block

```python
exp_scores = torch.exp(scores - new_max)
```

These are the current block’s **unnormalized** attention weights.

```python
acc = acc * correction + torch.bmm(exp_scores, vj)
```

This updates the weighted-value numerator:

1. Rescale the old numerator.
2. Add the current block’s weighted values.

```python
row_sum = row_sum * correction + exp_scores.sum(dim=-1, keepdim=True)
```

This updates the softmax denominator:

1. Rescale the old denominator.
2. Add the current block’s exponential sum.

Finally:

```python
row_max = new_max
```

The updated maximum is carried into the next iteration.

On the first iteration, `row_max` is `-inf`, so for finite scores:

```python
correction = exp(-inf) = 0
```

This makes the initial update work naturally.

### The invariant

After any number of key blocks, for each query the variables represent:

\[
\text{row\_sum}
=
\sum_{\text{keys seen}} e^{s_t-m}
\]

\[
\text{acc}
=
\sum_{\text{keys seen}} e^{s_t-m}v_t
\]

where \(m=\text{row\_max}\).

That invariant is what allows the function to compute full-sequence attention without storing all the scores.

---

## 6. Normalize and write the result

Once every key block has been processed:

```python
output[:, i:i+block_size] = acc / row_sum
```

For each query, this gives:

\[
\frac{\sum_t e^{s_t-m}v_t}
     {\sum_t e^{s_t-m}}
=
\sum_t \operatorname{softmax}(s)_t\,v_t
\]

So the result is mathematically the same as ordinary attention, up to floating-point rounding.

The outer loop repeats this process for the remaining query blocks.

---

## Practical caveats

- **It saves intermediate memory, not asymptotic computation.** It still evaluates every query–key pair, so computation remains approximately \(O(BS^2D)\).
- **It is FlashAttention-inspired, not an optimized FlashAttention implementation.** Actual FlashAttention uses fused GPU kernels and carefully manages GPU memory access. Python loops and separate PyTorch operations do not provide those same speed benefits.
- **It is not causal attention.** There is no mask, so each position can attend to future positions as well as past ones.
- **There is no explicit multi-head dimension.** The function operates on three-dimensional tensors; multiple heads would need to be handled separately or folded into the batch dimension.
- **Mixed precision needs care.** `row_max`, `row_sum`, and `acc` use PyTorch’s default floating-point dtype, typically `float32`. With `float16` or `bfloat16` inputs, this can cause a dtype mismatch in `torch.bmm(exp_scores, vj)`.
- **Training memory is a separate concern.** Standard autograd may retain intermediates from the loop. Actual FlashAttention also uses a memory-efficient backward implementation.

**In short:** the function computes ordinary attention block by block, maintaining a running maximum, denominator, and weighted-value numerator so that the final result is correct without constructing the full attention matrix.