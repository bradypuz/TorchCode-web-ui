# Mixture of Experts (MoE)

## Goal and intuition

Give each token access to several feed-forward networks, then route it to only a selected few. A learned router scores the experts for each token. The top-scoring experts produce outputs, and their normalized scores determine the weighted mixture. This increases the available parameter pool while limiting how many experts process each token.

## Inputs and outputs

- Input `x`: `(B, S, d_model)`, or a 2-D token matrix `(N, d_model)`.
- Router logits: `(N, num_experts)` after flattening a sequence batch.
- Selected expert IDs and weights: `(N, top_k)`.
- Each expert maps `d_model -> d_ff -> d_model`.
- Output: the original input shape.

The intended range is `1 <= top_k <= num_experts`.

## Walk through the reference

1. Create a linear router and a `ModuleList` of independent `Linear -> ReLU -> Linear` experts.
2. Save the input shape. Flatten `(B, S, D)` into `(B*S, D)` so routing treats every token independently.
3. Compute router logits, select their top `top_k` entries, and apply softmax to those **selected logits**.
4. Allocate a zero output with the flattened input's shape.
5. For each selection slot and each expert, build a Boolean mask identifying its assigned tokens. Run the expert only on those tokens.
6. Multiply each expert result by its routing weight, add it into the appropriate output rows, and restore the original shape.

## Why it works

For token $t$ and its selected expert set $\mathcal{T}_t$:

$$
y_t=\sum_{e\in\mathcal{T}_t}\frac{e^{g_{t,e}}}{\sum_{j\in\mathcal{T}_t}e^{g_{t,j}}}E_e(x_t)
$$

Weights sum to one within the selected set. Gradients can reach the selected experts and selected router scores. The discrete selection itself is not differentiable, so unselected scores receive no direct gradient through this mixture.

## Pitfalls and reference scope

This reference normalizes **after top-k**, rather than computing a full-expert softmax and leaving selected weights unnormalized. With `top_k=1`, softmax returns one, making the router's output weight constant and its gradient zero; stronger routing-learning mechanisms are absent. There is no load-balancing loss, expert capacity, token dropping, distributed dispatch, or fused routing implementation. Python loops and `mask.any()` prioritize clarity over speed. Arguments are not validated, and the flattening logic is intended for 2-D or 3-D inputs. Tests check structure, router/output shapes, and input gradient flow, but do not enforce balanced expert use.
