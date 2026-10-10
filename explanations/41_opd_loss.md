# On-Policy Distillation Reverse-KL Loss

## Intuition

Train a student to agree with one or more teachers on contexts visited by the student. This objective measures how much probability the student places on tokens the teachers consider unlikely. It computes a reverse KL from student to each teacher and combines those divergences with normalized teacher weights.

## Inputs and output

`student_logits` has shape `(..., V)`. One teacher has the same shape; multiple teachers add a leading axis, `(T, ..., V)`. Optional `teacher_weights` has shape `(T,)`, and optional `mask` has shape `(...)`, with one for included positions and zero for padding. The output is a scalar tensor. `temperature` is a scalar, defaulting to one.

## Implementation walkthrough

1. Add a leading teacher axis for the single-teacher case. Reject teacher ranks other than equal to the student's rank or one greater.
2. Detach teacher logits so teacher targets remain frozen. Divide both sets of logits by temperature and compute `log_softmax` over vocabulary.
3. Exponentiate student log-probabilities to get student probabilities. Broadcast them across teachers and sum `student_prob * (student_logp - teacher_logp)` over vocabulary.
4. Use uniform teacher weights when absent; otherwise move weights to the KL tensor's dtype/device and divide by their sum.
5. Broadcast weights across token positions and sum the teacher axis. Average all positions or use the mask-weighted sum divided by `mask.sum().clamp_min(1.0)`.
6. Multiply the resulting loss by `temperature ** 2`.

## Why it works

For temperature $\tau$, student probabilities $p$, and teacher probabilities $q_j$:

$$
K_j(x)=D_{\mathrm{KL}}(p\|q_j)=\sum_v p(v\mid x)\left[\log p(v\mid x)-\log q_j(v\mid x)\right]
$$

$$
\mathcal L=\tau^2\operatorname{Mean}_{x}\left[\sum_jw_jK_j(x)\right]
$$

The direction is student-to-teacher: student probabilities weight the token terms. This is the **weighted mean of separate KL divergences**, not KL against an arithmetic mixture of teacher probabilities. Averaging teacher logits first would also change the objective. Temperatures above one soften distributions; below one, they sharpen them. The squared multiplier rescales the training signal as specified by this task.

## Pitfalls and reference scope

Keep vocabulary reduction separate from teacher and token reductions. Student probabilities must remain differentiable. Teachers are detached, but supplied teacher weights are not explicitly detached. An all-zero binary mask gives zero loss for finite inputs. The reference checks teacher rank, without fully validating dimension compatibility, positive temperature, nonnegative weights, or a nonzero weight sum. Invalid values can still produce NaNs. It receives logits for existing contexts; it does not generate on-policy trajectories. The judge verifies direction, weighted teachers, masking, frozen teacher gradients, and temperature scaling.
