# Linear Self-Attention

The goal is to produce a weighted average of values without building the large table of every query–key comparison. This reference changes the similarity rule: it uses a feature-map dot product instead of softmax. Its speed comes from collecting a compact key/value summary before looking at the queries.

## First, identify the dimensions

`B` is batch size, `S` is the number of tokens, `D_k` is the query/key feature width, and `D_v` is the value width. Inputs `Q` and `K` have shape `(B, S, D_k)`; `V` has shape `(B, S, D_v)`. The result has shape `(B, S, D_v)`: each query receives one value-sized answer. Batch items are independent.

Think of `K` as descriptions of available items and `V` as information attached to those items. A query should put more weight on items whose key features agree with its own features.

## 1. Turn queries and keys into non-negative features

```python
Q_prime = F.elu(Q) + 1
K_prime = F.elu(K) + 1
```

`F.elu` acts on every element and preserves shape. With its default parameter, adding one gives the feature map

$$
\phi(x)=\begin{cases}x+1,&x\geq 0\\e^x,&x<0.\end{cases}
$$

For instance, an input coordinate `0` becomes `1`, and `2` becomes `3`. A negative coordinate becomes a small positive number in exact arithmetic. The dot product of two such feature vectors is therefore a non-negative similarity that can be normalized into weights. These features are not probabilities by themselves; their coordinates do not need to sum to one.

## 2. Summarize keys and values before processing queries

```python
KV = torch.bmm(K_prime.transpose(1, 2), V)
Z = K_prime.sum(dim=1, keepdim=True)
```

`transpose(1, 2)` changes the key layout from `(B, S, D_k)` to `(B, D_k, S)`. `bmm` then contracts the shared sequence dimension with `V`, producing `(B, D_k, D_v)`. Every summary entry adds up a key-feature/value product across all tokens.

Take one batch with transformed keys `[[2, 1], [1, 3]]` and scalar values `[[10], [20]]`. The summary is `[[2*10 + 1*20], [1*10 + 3*20]] = [[40], [70]]`. Its size depends on the feature widths rather than the square of the token count.

`Z` sums the transformed keys over **tokens**, not over features. In this example it is `[[3, 4]]`. `keepdim=True` retains a size-one sequence axis, giving `(B, 1, D_k)`, ready for the next matrix product.

```text
transformed keys (B,S,D_k) ─ transpose → (B,D_k,S) ─ @ values → summary (B,D_k,D_v)
             └─────────── sum over tokens ─────────────────→ key total (B,1,D_k)

query features [1,1] @ summary [[40],[70]] → numerator 110
query features [1,1] · key total [3,4]     → denominator 7
                                                      ↓
                                            output ≈ 110/7
```

## 3. Read the summary and compute the matching denominator

```python
num = torch.bmm(Q_prime, KV)
den = torch.bmm(Q_prime, Z.transpose(1, 2))
return num / (den + 1e-6)
```

A transformed query `[1, 1]` reads `1*40 + 1*70 = 110` from the summary. Its denominator is `1*3 + 1*4 = 7`. Why seven? That query's similarities with the two keys are `3` and `4`; seven is their sum. Its output is approximately `(3*10 + 4*20)/7 = 15.7143`.

The numerator has shape `(B, S, D_v)`. Transposing `Z` gives `(B, D_k, 1)`, so the denominator has shape `(B, S, 1)`. Division broadcasts each query's single denominator across all of its value coordinates. `1e-6` protects very small denominators and slightly changes the exact normalized result.

## Run a worked example

Run this after loading the reference code. The assertions check both the intermediate calculations and the function's returned answers.

```python
# Example: Two queries read the same key-value summary
Q = torch.tensor([[[0., 0.], [1., 0.]]], requires_grad=True)
K = torch.tensor([[[1., 0.], [0., 2.]]], requires_grad=True)
V = torch.tensor([[[10.], [20.]]], requires_grad=True)
q_features = F.elu(Q) + 1
k_features = F.elu(K) + 1
summary = torch.bmm(k_features.transpose(1, 2), V)
key_sum = k_features.sum(dim=1, keepdim=True)
denominator = torch.bmm(q_features, key_sum.transpose(1, 2))
assert torch.allclose(summary, torch.tensor([[[40.], [70.]]]))
assert torch.allclose(denominator, torch.tensor([[[7.], [10.]]]))
out = linear_attention(Q, K, V)
expected = torch.tensor([[[110. / (7. + 1e-6)], [150. / (10. + 1e-6)]]])
assert out.shape == (1, 2, 1)
assert torch.allclose(out, expected)
out.sum().backward()
assert Q.grad is not None and K.grad is not None and V.grad is not None
```

The second transformed query is `[2, 1]`, which has similarity `5` with each key. It therefore returns approximately the simple average `15`. Different queries use the same summary but obtain different weighted answers.

## Why this saves work, and what it does not do

Associativity lets the code calculate `phi(Q) @ (phi(K).T @ V)` without first calculating `phi(Q) @ phi(K).T`. The main work is `O(B*S*D_k*D_v)` for fixed feature widths. This computes kernel attention, not the exact softmax result. Its summary includes every token, so it is noncausal and has no padding mask. Autograd can reach all three inputs through the tensor operations. Very negative features can round to zero in finite precision, and the reference does not explicitly upcast accumulation.
