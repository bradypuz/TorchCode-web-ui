This implements **linear attention**: each query produces a weighted average of the values, using its similarity to the keys—but without constructing the full query–key attention matrix.

The main trick is to **summarize the keys and values first**, then let each query read from that summary.

## 1. What are `Q`, `K`, and `V`?

In attention:

- **Queries (`Q`)** describe what each position is looking for.
- **Keys (`K`)** describe what each position offers for matching.
- **Values (`V`)** contain the information that gets combined into the output.

For a self-attention sequence of length `S`, the shapes are:

| Tensor | Shape | Meaning |
|---|---|---|
| `Q` | `(B, S, D_k)` | One query vector per position |
| `K` | `(B, S, D_k)` | One key vector per position |
| `V` | `(B, S, D_v)` | One value vector per position |

Here, `B` is the batch size, `D_k` is the query/key dimension, and `D_v` is the value dimension.

The output has shape **`(B, S, D_v)`**.

## 2. Transform queries and keys into positive features

```python
Q_prime = F.elu(Q) + 1
K_prime = F.elu(K) + 1
```

This applies a feature map, element by element:

\[
\phi(x)=\operatorname{ELU}(x)+1
=
\begin{cases}
x+1, & x>0\\
e^x, & x\leq 0
\end{cases}
\]

So, mathematically, the transformed features are positive.

The similarity between query \(i\) and key \(j\) is then:

\[
s_{ij}=\phi(q_i)^\top\phi(k_j)
\]

Because the features are positive, these similarities can be used as nonnegative weights in a weighted average.

**Important:** this is a different similarity rule from standard softmax attention. It is not computing `softmax(Q @ K.T)`.

## 3. Build a compact key–value summary

```python
KV = torch.bmm(K_prime.transpose(1, 2), V)
```

`torch.bmm` performs matrix multiplication separately for each batch item.

The shapes are:

```text
K_prime.transpose(1, 2): (B, D_k, S)
V:                       (B, S, D_v)
KV:                      (B, D_k, D_v)
```

For one batch item, this computes:

\[
KV=\sum_{j=1}^{S}\phi(k_j)v_j^\top
\]

Each key–value pair contributes an outer product to a shared summary.

**Intuition:** `KV` captures which value information is associated with each key-feature dimension. Its size does not grow with sequence length.

Next:

```python
Z = K_prime.sum(dim=1, keepdim=True)
```

This sums the keys across all positions:

\[
Z=\sum_{j=1}^{S}\phi(k_j)
\]

Its shape is `(B, 1, D_k)`. This summary will be used to normalize the output.

## 4. Let each query read from the summaries

### Numerator: weighted value sum

```python
num = torch.bmm(Q_prime, KV)
```

The shapes are:

```text
Q_prime: (B, S, D_k)
KV:      (B, D_k, D_v)
num:     (B, S, D_v)
```

For query \(i\):

\[
\text{num}_i
=
\phi(q_i)^\top
\left(\sum_j \phi(k_j)v_j^\top\right)
=
\sum_j
\underbrace{\left(\phi(q_i)^\top\phi(k_j)\right)}_{\text{similarity weight}}
v_j
\]

So this is the sum of all values, weighted by their keys’ similarity to the query.

### Denominator: total weight

```python
den = torch.bmm(Q_prime, Z.transpose(1, 2))
```

The shapes are:

```text
Q_prime:          (B, S, D_k)
Z.transpose(1, 2): (B, D_k, 1)
den:              (B, S, 1)
```

For query \(i\):

\[
\text{den}_i
=
\phi(q_i)^\top\sum_j\phi(k_j)
=
\sum_j\phi(q_i)^\top\phi(k_j)
\]

This is the total similarity weight assigned by that query.

### Normalize

```python
return num / (den + 1e-6)
```

The denominator broadcasts across the `D_v` output features. The small `1e-6` stabilizes division when the denominator is very small.

Putting everything together:

\[
\boxed{
o_i=
\frac{
\sum_j \left(\phi(q_i)^\top\phi(k_j)\right)v_j
}{
\sum_j \phi(q_i)^\top\phi(k_j)+10^{-6}
}
}
\]

Ignoring the small epsilon, this is a **weighted average of the values**, with weights that sum to 1.

## 5. Why is it called “linear” attention?

You could calculate the same result by explicitly forming:

```python
scores = torch.bmm(Q_prime, K_prime.transpose(1, 2))  # (B, S, S)
num = torch.bmm(scores, V)
den = scores.sum(dim=-1, keepdim=True)
```

But that creates an `S × S` matrix: every query is compared with every key.

Your implementation instead uses the associativity of matrix multiplication:

\[
\underbrace{(Q'K'^\top)}_{S\times S}V
=
Q'\underbrace{(K'^\top V)}_{D_k\times D_v}
\]

It computes the right-hand side, avoiding the full attention matrix.

For fixed feature dimensions:

- **Standard dense attention:** quadratic cost in sequence length, roughly \(O(S^2)\).
- **This implementation:** linear cost in sequence length, roughly \(O(SD_kD_v)\).

The largest attention-specific summary is `D_k × D_v`, rather than `S × S`.

### Two important limitations

- **It is not equivalent to standard softmax attention.** It uses normalized feature-map dot products instead.
- **It is not causal.** Every query uses every key/value in the input, including later positions. Autoregressive attention would need prefix summaries so each position sees only itself and earlier positions.

**In short:** transform queries and keys into positive features, compress all key–value pairs into a shared summary, then use each query to retrieve and normalize information from that summary.