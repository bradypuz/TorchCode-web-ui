This function computes a **distillation loss that encourages a student model’s probability distribution to match one or more teacher models**.

It uses **reverse KL divergence**:

\[
D_{\mathrm{KL}}(p_{\text{student}}\|p_{\text{teacher}})
\]

The student receives gradients; the teacher logits are treated as fixed targets.

## 1. What the inputs represent

For a language model, a typical shape is:

```python
student_logits.shape = (batch_size, sequence_length, vocabulary_size)
```

Each token position has a vector of scores over the vocabulary. These are **logits**, not probabilities.

The teacher can provide either:

```python
# One teacher
teacher_logits.shape = (batch_size, sequence_length, vocabulary_size)

# Multiple teachers
teacher_logits.shape = (
    num_teachers, batch_size, sequence_length, vocabulary_size
)
```

The `...` notation in the docstring means the function also supports other leading dimensions.

The other inputs are:

- **`teacher_weights`**: How much each teacher contributes.
- **`mask`**: Which token positions contribute to the final loss.
- **`temperature`**: How sharp or smooth the student and teacher distributions are.

---

## 2. Give teacher logits a consistent shape

```python
if teacher_logits.dim() == student_logits.dim():
    teacher_logits = teacher_logits.unsqueeze(0)
elif teacher_logits.dim() != student_logits.dim() + 1:
    raise ValueError(...)
```

If there is only one teacher, the code adds a leading dimension of size `1`.

For example:

```text
Before: (B, L, V)
After:  (1, B, L, V)
```

Now both single-teacher and multi-teacher cases can use the same code, with the first dimension always indexing teachers.

This checks the **number of dimensions**, but does not fully validate that all dimension sizes match.

## 3. Stop gradients from flowing into the teacher

```python
teacher_logits = teacher_logits.detach()
t = float(temperature)
```

`.detach()` removes the teacher logits from the gradient computation graph.

When you call:

```python
loss.backward()
```

the loss can update the student, but it will not backpropagate through these teacher logits.

Converting the temperature to a Python `float` also means the temperature is treated as a fixed value, not a trainable tensor.

## 4. Convert logits into distributions

```python
student_logp = F.log_softmax(student_logits / t, dim=-1)
student_prob = student_logp.exp()
teacher_logp = F.log_softmax(teacher_logits / t, dim=-1)
```

For student logits \(s_v\), this computes:

\[
p_s(v)=\frac{\exp(s_v/t)}{\sum_u \exp(s_u/t)}
\]

and similarly for each teacher.

The resulting tensors contain:

| Variable | Meaning |
|---|---|
| `student_logp` | \(\log p_s(v)\) |
| `student_prob` | \(p_s(v)\) |
| `teacher_logp` | \(\log p_{t,j}(v)\) for teacher \(j\) |

`F.log_softmax` is a numerically stable way to compute log probabilities directly.

### What temperature does

- **`t = 1`**: Ordinary softmax.
- **`t > 1`**: Smoother distributions; lower-ranked vocabulary items get more probability.
- **`0 < t < 1`**: Sharper distributions; high-scoring items dominate more strongly.

Both student and teacher use the same temperature.

## 5. Compute reverse KL for every teacher and token

```python
per_teacher_kl = (
    student_prob.unsqueeze(0) *
    (student_logp.unsqueeze(0) - teacher_logp)
).sum(dim=-1)
```

This is the core calculation:

\[
D_{\mathrm{KL}}(p_s\|p_{t,j})
=
\sum_v p_s(v)
\left[\log p_s(v)-\log p_{t,j}(v)\right]
\]

The sum is over vocabulary items.

For the language-model example:

```text
student_prob.unsqueeze(0): (1, B, L, V)
teacher_logp:              (T, B, L, V)
per_teacher_kl:            (T, B, L)
```

Broadcasting compares the same student distribution against every teacher.

### Why “reverse” KL?

In distillation, the more common direction is:

\[
D_{\mathrm{KL}}(p_{\text{teacher}}\|p_{\text{student}})
\]

This code uses the opposite direction, so the expectation is weighted by **student probabilities**.

Intuitively, it especially penalizes the student for putting probability on outputs the teacher considers unlikely. Reverse KL is often described as **mode-seeking**: when the teacher has several plausible modes, the student may concentrate on a subset rather than cover them all.

Importantly, `student_prob` is **not detached**. Gradients flow through both the student probabilities and student log probabilities.

## 6. Combine multiple teachers

First, the function chooses teacher weights:

```python
if teacher_weights is None:
    weights = torch.full(
        (num_teachers,),
        1.0 / num_teachers,
        ...
    )
else:
    weights = teacher_weights.to(...)
    weights = weights / weights.sum()
```

Without explicit weights, every teacher contributes equally.

If you pass:

```python
teacher_weights = torch.tensor([2.0, 1.0])
```

the normalized weights become:

```text
[2/3, 1/3]
```

Then:

```python
view_shape = (num_teachers,) + (1,) * (per_teacher_kl.dim() - 1)
per_token_kl = (weights.view(view_shape) * per_teacher_kl).sum(dim=0)
```

reshapes the weights for broadcasting and computes:

\[
\ell_i=\sum_j w_j D_{\mathrm{KL}}(p_{s,i}\|p_{t,j,i})
\]

For `(T, B, L)` divergences, weights are reshaped to `(T, 1, 1)`. The resulting `per_token_kl` has shape `(B, L)`.

**A subtle distinction:** this averages the teachers’ **KL losses**, not their probability distributions. It is not the same as first computing an arithmetic mixture of teacher probabilities and then taking KL against that mixture. For valid normalized weights, its student-dependent part is equivalent to matching a normalized **weighted geometric mean** of the teacher distributions.

## 7. Average over included token positions

With a mask:

```python
mask = mask.to(dtype=per_token_kl.dtype, device=per_token_kl.device)
loss = (per_token_kl * mask).sum() / mask.sum().clamp_min(1.0)
```

A binary mask such as:

```text
[1, 1, 0, 0]
```

means “include the first two positions, ignore the last two.”

The calculation is:

\[
\text{loss}
=
\frac{\sum_i m_i\ell_i}
{\max(\sum_i m_i,1)}
\]

This is useful for excluding padding tokens or, when desired, prompt tokens.

The denominator clamp prevents division by zero. If every mask entry is zero and the per-token losses are finite, the result is zero.

Without a mask:

```python
loss = per_token_kl.mean()
```

all positions contribute equally.

## 8. Apply temperature scaling

```python
return loss * (t ** 2)
```

Multiplying by \(t^2\) is a conventional distillation adjustment intended to compensate for changes in gradient scale introduced by temperature.

It does **not** make different temperatures produce exactly the same loss or gradients.

## Overall formula

With a binary mask, the function computes:

\[
\boxed{
L=t^2
\frac{
\sum_i m_i
\sum_j w_j
\sum_v p_{s,i}(v)
\log\frac{p_{s,i}(v)}{p_{t,j,i}(v)}
}{
\max(\sum_i m_i,1)
}
}
\]

In plain English:

> At each included token position, compare the student distribution with every teacher using reverse KL, combine the comparisons using teacher weights, average across positions, and apply temperature scaling.

### Two practical caveats

1. **Inputs need validation.** Temperature should be positive; teacher weights should have shape `(T,)`, be nonnegative, and have a positive sum. The code does not enforce these conditions.

2. **“On-policy” depends on the surrounding training loop.** This function does not generate student samples. It is on-policy distillation only if the logits are evaluated on contexts or trajectories generated by the current student policy. By itself, it is a reverse-KL distillation loss.