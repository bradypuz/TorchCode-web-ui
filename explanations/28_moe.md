This implements a **sparse Mixture of Experts (MoE) layer**: instead of sending every input through one shared feed-forward network, it has several networks called *experts*. A *router* selects a few experts for each input vector, and their outputs are combined using learned weights.

With the default `top_k=2`, **each input vector goes through two experts**.

## 1. What gets created?

```python
def __init__(self, d_model, d_ff, num_experts, top_k=2):
```

The parameters mean:

| Parameter | Meaning |
|---|---|
| `d_model` | Size of each input and output vector |
| `d_ff` | Hidden-layer size inside each expert |
| `num_experts` | Total number of available experts |
| `top_k` | Number of experts selected for each input vector |

### The router

```python
self.router = nn.Linear(d_model, num_experts)
```

For each input vector of length `d_model`, the router produces one score for each expert:

```text
Input:          [d_model]
Router scores:  [num_experts]
```

These scores are called **logits**. Higher scores indicate which experts the router prefers for that input.

### The experts

```python
self.experts = nn.ModuleList([
    nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.ReLU(),
        nn.Linear(d_ff, d_model)
    )
    for _ in range(num_experts)
])
```

Each expert is an independent feed-forward network:

```text
d_model → d_ff → ReLU → d_model
```

They all have the same architecture but **different trainable parameters**. `nn.ModuleList` registers them with PyTorch, so their parameters are included in training and operations such as `model.to(device)`.

## 2. Flatten the input

```python
orig_shape = x.shape
if x.dim() == 3:
    B, S, D = x.shape
    x_flat = x.reshape(-1, D)
else:
    x_flat = x
```

The code is intended to accept either:

- **`[B, S, D]`**: a batch of sequences, where `B` is batch size, `S` is sequence length, and `D = d_model`.
- **`[N, D]`**: a collection of input vectors.

For a three-dimensional input, it combines the batch and sequence dimensions:

```text
[B, S, D] → [B × S, D]
```

This lets the layer route **each token independently**. It does not mix information between tokens as attention does.

The original shape is saved so it can be restored afterward.

## 3. Score all experts and select the best ones

```python
logits = self.router(x_flat)
top_vals, top_idx = logits.topk(self.top_k, dim=-1)
```

Let `N` be the number of flattened input vectors and `E` the number of experts:

```text
logits:    [N, E]
top_vals:  [N, top_k]
top_idx:   [N, top_k]
```

For every input vector:

- `top_vals` contains the highest `top_k` router scores.
- `top_idx` contains the corresponding expert indices.

For example, suppose an input receives these router scores:

```text
Expert:   0    1    2     3
Score:   0.1  2.0  1.4  -0.5
```

With `top_k=2`, the router selects:

```text
top_idx  = [1, 2]
top_vals = [2.0, 1.4]
```

Only experts 1 and 2 will process that input.

## 4. Turn selected scores into mixing weights

```python
weights = torch.softmax(top_vals, dim=-1)
```

Softmax converts the selected scores into positive weights that sum to 1 **across the selected experts**.

For the example above:

```text
softmax([2.0, 1.4]) ≈ [0.646, 0.354]
```

The final output will therefore be:

```text
0.646 × expert_1(input) + 0.354 × expert_2(input)
```

Notice that softmax is applied **after selecting the top experts**, not over all experts.

## 5. Run the selected experts and accumulate their outputs

```python
output = torch.zeros_like(x_flat)
```

This initializes an output buffer of shape `[N, D]`.

Then:

```python
for k in range(self.top_k):
    for e in range(len(self.experts)):
        mask = (top_idx[:, k] == e)
        if mask.any():
            output[mask] += (
                weights[mask, k:k+1]
                * self.experts[e](x_flat[mask])
            )
```

Here:

- `k` is the **selection position**: first-choice expert, second-choice expert, etc.
- `e` is an **expert index**.
- `mask` identifies all inputs whose expert at position `k` is expert `e`.

For example:

```python
mask = (top_idx[:, 0] == 1)
```

selects all input vectors whose first-choice expert is expert 1.

For those inputs:

1. `x_flat[mask]` gathers the input vectors.
2. `self.experts[e](...)` processes them together.
3. `weights[mask, k:k+1]` supplies their mixing weights.
4. The weighted results are added to `output`.

The slice `k:k+1` preserves a dimension:

```text
Selected inputs:  [M, D]
Expert outputs:   [M, D]
Mixing weights:   [M, 1]
```

That `[M, 1]` shape allows broadcasting: each input’s scalar weight multiplies every feature of its expert output.

Mathematically, for one input vector \(x_i\):

\[
y_i = \sum_{k=1}^{K} w_{i,k}\,f_{e_{i,k}}(x_i)
\]

where \(e_{i,k}\) is its selected expert, \(f_e\) is expert \(e\), and \(w_{i,k}\) is the corresponding mixing weight.

## 6. Restore the original shape

```python
return output.reshape(orig_shape)
```

The layer preserves the input shape:

```text
[B, S, D] → MoE → [B, S, D]
```

or:

```text
[N, D] → MoE → [N, D]
```

## Important details

- **Sparse expert computation:** The router scores every expert, but each input is processed by only `top_k` experts.
- **Learned routing:** Gradients can flow through the selected scores and their softmax weights. The discrete choice of expert indices is not differentiable.
- **Special case: `top_k=1`:** Softmax over one score is always `1`, so this implementation provides no task-loss gradient to the router through the mixing weights.
- **No load-balancing mechanism:** Nothing here explicitly encourages even expert usage. Some experts may receive many inputs while others receive few.
- **Simple rather than optimized:** The nested loops and boolean indexing make the logic easy to follow, but large production MoE implementations typically use more efficient dispatching.
- **Input assumptions:** This expects two- or three-dimensional inputs with last dimension `d_model`, and `1 <= top_k <= num_experts`.

**In short:** for each input vector, the layer chooses a small set of feed-forward networks, runs those networks, and returns a weighted combination of their outputs.