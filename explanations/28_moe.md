# Mixture of Experts

## What the layer does

For each input token, choose a few feed-forward networks, called **experts**, and combine only their results. A small learned **router** assigns scores to the experts. Different tokens can choose different experts, so the layer need not run every expert on every token.

`x` has shape `(B, S, D)` for sequences or `(N, D)` for a batch of vectors. `B` is the batch size, `S` the sequence length, `N` the number of vectors, and `D = d_model` the feature width. Each expert's internal width is `d_ff`. With `E = num_experts` and `K = top_k`, each token selects `K` of the `E` experts. The output has exactly the original input shape.

## Register the router and expert networks

```python
self.top_k = top_k
self.router = nn.Linear(d_model, num_experts)
self.experts = nn.ModuleList([
    nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
    for _ in range(num_experts)
])
```

After the parent module initializer, these statements store the selection count and create a router producing `E` logits per token. Each expert expands `D` features to `d_ff`, replaces negative intermediate values with zero using ReLU, and projects back to `D`. All linear layers have default biases.

`nn.ModuleList` registers every expert as part of the containing module. It does not itself run them; the later dispatch loops decide which tokens each expert receives. Registration ensures their parameters are visible to optimizers and module device transfers.

## Flatten tokens, then select and normalize their expert scores

```python
orig_shape = x.shape
if x.dim() == 3:
    B, S, D = x.shape
    x_flat = x.reshape(-1, D)
else:
    x_flat = x
logits = self.router(x_flat)
top_vals, top_idx = logits.topk(self.top_k, dim=-1)
weights = torch.softmax(top_vals, dim=-1)
```

Saving `orig_shape` permits restoring the original batch/sequence arrangement at the end. A 3-D input is flattened into `(B * S, D)`, so dispatch can work with one list of tokens. A 2-D input is already in this form.

Router logits have shape `(N, E)`. `topk` returns the selected scores and their expert indices, each of shape `(N, K)`. Slot zero is the highest selected score; slot one is the next, and so on. Softmax is applied to the **selected scores**, giving selected-expert weights summing to one for each token. Unselected experts receive no contribution. This is equivalent to renormalizing the selected probabilities from a full softmax, but the code never needs to compute that full softmax.

## Dispatch subsets of tokens and accumulate their outputs

```python
output = torch.zeros_like(x_flat)
for k in range(self.top_k):
    for e in range(len(self.experts)):
        mask = (top_idx[:, k] == e)
        if mask.any():
            output[mask] += weights[mask, k:k+1] * self.experts[e](x_flat[mask])
return output.reshape(orig_shape)
```

The loops visit one selection slot `k` and one expert `e` at a time. `mask` says which tokens selected this expert in this slot. If no token did, the expert is skipped. Otherwise, `x_flat[mask]` gathers only those tokens and sends them through that expert.

`weights[mask, k:k+1]` has shape `(selected_tokens, 1)`. Keeping that last axis allows each token's scalar gate to multiply all `D` output features through broadcasting. `+=` adds the contribution because a token can receive outputs from several selected experts. The final reshape restores the sequence layout.

```text
Tokens (N, D) → router scores (N, E) → chosen IDs and gates (N, K)
    │                                      │
    ├→ selected tokens for expert 0 ─→ weighted results ─┐
    ├→ selected tokens for expert 1 ─→ weighted results ─┼→ sum per token
    └→ selected tokens for expert 2 ─→ weighted results ─┘
```

## Trace two tokens through three experts

Use one-feature positive tokens `1` and `2`. Configure the experts to return `x`, `3*x`, and `5*x`. Configure the router to score them as `[3 - x, 2*x - 1, 0]`, and select two experts.

| Token | Router logits | Selected IDs | Normalized gates | Selected expert outputs | Final result |
|---|---|---|---|---|---|
| `1` | `[2, 1, 0]` | `[0, 1]` | `[0.7311, 0.2689]` | `[1, 3]` | `1.5379` |
| `2` | `[1, 3, 0]` | `[1, 0]` | `[0.8808, 0.1192]` | `[6, 2]` | `5.5232` |

In slot zero, expert zero receives token `1` and expert one receives token `2`. In slot one, those assignments swap. Expert two receives no tokens at all. Both tokens still obtain a sum of two weighted expert results.

```python
# Example: Dispatch two tokens to different expert pairs and sum their outputs
model = MixtureOfExperts(d_model=1, d_ff=1, num_experts=3, top_k=2)
with torch.no_grad():
    model.router.weight.copy_(torch.tensor([[-1.], [2.], [0.]]))
    model.router.bias.copy_(torch.tensor([3., -1., 0.]))
    for expert, multiplier in zip(model.experts, (1., 3., 5.)):
        expert[0].weight.fill_(1.)
        expert[0].bias.zero_()
        expert[2].weight.fill_(multiplier)
        expert[2].bias.zero_()
x = torch.tensor([[[1.], [2.]]], requires_grad=True)
logits = model.router(x.reshape(2, 1))
assert torch.equal(logits, torch.tensor([[2., 1., 0.], [1., 3., 0.]]))
assert torch.equal(logits.topk(2, dim=-1).indices, torch.tensor([[0, 1], [1, 0]]))
s = torch.sigmoid(torch.tensor(1.)).item()
t = torch.sigmoid(torch.tensor(2.)).item()
out = model(x)
assert out.shape == (1, 2, 1)
assert torch.allclose(out, torch.tensor([[[3 - 2*s], [2 + 4*t]]]), atol=1e-6)
out.sum().backward()
assert model.router.weight.grad is not None
assert model.experts[0][0].weight.grad is not None
assert model.experts[2][0].weight.grad is None
```

Gradients reach selected experts and the selected score values. The discrete expert indices do not have gradients; selection changes when scores change enough to alter their ordering.

## Why a top-one gate is always one

When `top_k = 1`, softmax receives one selected logit per token. `exp(score) / exp(score)` equals one, even if the score is low. The returned token is therefore exactly its selected expert's result, not that result multiplied by its full-router probability. The single-entry softmax also has zero derivative, so this output alone gives the router no score gradient.

```python
# Example: Normalize a single selected expert to a gate of one
model = MixtureOfExperts(d_model=1, d_ff=1, num_experts=2, top_k=1)
with torch.no_grad():
    model.router.weight.zero_()
    model.router.bias.copy_(torch.tensor([2., 1.]))
    for expert, multiplier in zip(model.experts, (1., 3.)):
        expert[0].weight.fill_(1.)
        expert[0].bias.zero_()
        expert[2].weight.fill_(multiplier)
        expert[2].bias.zero_()
x = torch.tensor([[2.]])
out = model(x)
assert torch.equal(out, torch.tensor([[2.]]))
out.sum().backward()
assert torch.equal(model.router.bias.grad, torch.zeros(2))
assert torch.equal(model.router.weight.grad, torch.zeros(2, 1))
assert model.experts[1][0].weight.grad is None
```

## Common mistakes and reference scope

Normalize only the selected logits, preserve the gate's singleton feature axis, and add rather than overwrite contributions. Require `1 <= top_k <= num_experts`. The reference has no expert-capacity limit, token dropping, routing-noise mechanism, or load-balancing loss, and its nested Python loops illustrate dispatch rather than optimizing it.
