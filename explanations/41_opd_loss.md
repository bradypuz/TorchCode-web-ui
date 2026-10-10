# On-Policy Distillation Reverse-KL Loss

Train a student to agree with teachers on contexts supplied by the caller. This function computes a divergence from the student's distribution to each teacher's distribution, combines those divergences with teacher weights, and averages included token positions.

## Inputs and output

`student_logits` has shape `(..., V)`, where the last axis is vocabulary. A single teacher has the same shape; multiple teachers add a leading axis `(T, ..., V)`. `teacher_weights`, if supplied, has shape `(T,)`. A mask has shape `(...)`, with one for included positions and zero for ignored positions. The return is a scalar tensor.

## Reading the actual code

```python
if teacher_logits.dim() == student_logits.dim():
    teacher_logits = teacher_logits.unsqueeze(0)
elif teacher_logits.dim() != student_logits.dim() + 1:
    raise ValueError('teacher_logits must have shape (..., V) or (T, ..., V)')
teacher_logits = teacher_logits.detach()
t = float(temperature)
```

Adding a singleton teacher axis lets the rest of the code handle one or many teachers identically. Other ranks are rejected. Detaching teacher logits freezes teacher targets: gradients should change the student, not move the teachers toward it. Temperature is converted to a Python number.

```python
student_logp = F.log_softmax(student_logits / t, dim=-1)
student_prob = student_logp.exp()
teacher_logp = F.log_softmax(teacher_logits / t, dim=-1)
```

`log_softmax` normalizes along vocabulary and returns stable log-probabilities. Exponentiating student log-probabilities supplies the weights in reverse KL. Temperatures above one soften distributions; positive temperatures below one sharpen them. Neither student tensor is detached, because both the probabilities and their logarithms participate in learning.

```python
per_teacher_kl = (
    student_prob.unsqueeze(0) * (student_logp.unsqueeze(0) - teacher_logp)
).sum(dim=-1)
```

This is `KL(student || teacher)`: sum over vocabulary of `p_student * log(p_student/p_teacher)`. Reversing the probability weighting would compute a different direction. For student shape `(B,L,V)` and teachers `(T,B,L,V)`, adding a teacher axis gives student shape `(1,B,L,V)`. Broadcasting compares the same student with every teacher, and summing vocabulary produces `(T,B,L)`.

Consider student probabilities `[0.5,0.5]`. Teacher one agrees exactly, giving KL zero. Teacher two uses `[0.9,0.1]`, giving `0.5*log(0.5/0.9) + 0.5*log(0.5/0.1)`, approximately `0.510826`.

```python
weights = weights / weights.sum()
view_shape = (num_teachers,) + (1,) * (per_teacher_kl.dim() - 1)
per_token_kl = (weights.view(view_shape) * per_teacher_kl).sum(dim=0)
```

With supplied weights `[3,1]`, normalization gives `[0.75,0.25]`. The code first moves supplied weights to the KL tensor's dtype/device; without supplied weights it creates equal weights. Reshaping to `(T,1,1)` in our batched example broadcasts each teacher's weight across all token positions. Summing dimension zero removes the teacher axis.

| Token position | Student | Teacher 1 | Teacher 2 | KL to teacher 1 | KL to teacher 2 | Weighted KL |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 0 | `[0.5,0.5]` | `[0.5,0.5]` | `[0.9,0.1]` | 0 | 0.510826 | 0.127706 |
| 1 | `[0.8,0.2]` | `[0.8,0.2]` | `[0.2,0.8]` | 0 | 0.831777 | 0.207944 |

This averages **separate divergences**, not teacher probabilities. At position zero the arithmetic teacher mixture would be `[0.6,0.4]`, giving only about `0.020411` KL from the student, a different objective.

```python
loss = (per_token_kl * mask).sum() / mask.sum().clamp_min(1.0)
# Or, without a mask:
loss = per_token_kl.mean()
return loss * (t ** 2)
```

A mask `[1,0]` keeps only position zero, so loss is `0.127706` at temperature one. The denominator counts included positions rather than all positions. Clamping it prevents division by zero for an all-zero binary mask. The final temperature-squared factor rescales the softened loss as specified by the reference.

## Run weighted teachers and masking

Load the reference solution first, then run:

```python
# Example: Weight separate teacher divergences and ignore a padded position
student = torch.tensor([[0.5, 0.5], [0.8, 0.2]]).log().requires_grad_()
teachers = torch.tensor([[[0.5, 0.5], [0.8, 0.2]],
                         [[0.9, 0.1], [0.2, 0.8]]]).log().requires_grad_()
weights = torch.tensor([3.0, 1.0])
mask = torch.tensor([1.0, 0.0])
loss = opd_loss(student, teachers, weights, mask)
torch.testing.assert_close(loss, torch.tensor(0.1277064059))
unmasked = opd_loss(student, teachers, weights)
torch.testing.assert_close(unmasked, torch.tensor((0.1277064059 + 0.2079441542)/2))
loss.backward()
assert teachers.grad is None
assert torch.count_nonzero(student.grad[0]).item() > 0
assert torch.equal(student.grad[1], torch.zeros(2))
zero_mask = opd_loss(student, teachers, weights, torch.zeros(2))
assert zero_mask.item() == 0.0
```

## Common mistakes and reference limits

Preserve vocabulary, teacher, and position axes during their separate reductions. Supplied weights are normalized but not detached. The code checks teacher rank, not every dimension, positive temperature, nonnegative weights, or a nonzero weight sum. Finite inputs are needed for the all-zero-mask result. The function does not generate on-policy trajectories; it only evaluates supplied logits.
