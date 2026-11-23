"""Original-style GRPO loss with clipping, sampled-token KL, and masking."""

from textwrap import dedent


TASK = {
    "title": "GRPO Loss - Complete",
    "difficulty": "Hard",
    "function_name": "grpo_complete_loss",
    "hint": (
        "Normalize rewards across each prompt's group with population std clamped to 1e-6, "
        "detach the advantages, and broadcast them across tokens. Sanitize masked log-probs "
        "before exponentiating. Minimize the negative PPO-clipped surrogate plus "
        "beta * (expm1(ref_logps - policy_logps) - (ref_logps - policy_logps)). "
        "Average valid tokens within each response, then average all responses. "
        "Gradients should flow through valid policy_logps only."
    ),
    "tests": [
        {
            "name": "Scalar loss, dtype, device, and defaults",
            "code": dedent("""
                import math
                import torch
                policy = torch.full((1, 2, 3), -1., dtype=torch.float64, requires_grad=True)
                old = torch.full_like(policy, -1.)
                ref = torch.full_like(policy, -0.5)
                rewards = torch.tensor([[0., 2.]], dtype=policy.dtype)
                mask = torch.ones_like(policy, dtype=torch.bool)
                loss = {fn}(policy, old, ref, rewards, mask)
                expected = torch.tensor(0.04 * (math.expm1(0.5) - 0.5), dtype=policy.dtype)
                assert isinstance(loss, torch.Tensor) and loss.dim() == 0, 'Return a scalar Tensor'
                assert loss.dtype == policy.dtype and loss.device == policy.device, 'Preserve the policy dtype and device'
                assert loss.requires_grad, 'Keep the loss connected to policy_logps'
                assert torch.allclose(loss, expected, atol=1e-12, rtol=1e-12), 'The default beta is 0.04; opposite advantages cancel when ratios are one'
            """),
        },
        {
            "name": "Numeric check against a scalar token oracle",
            "code": dedent("""
                import math
                import torch
                ratios = torch.tensor([[[0.6, 1.1, 1.4], [0.7, 1., 1.3]], [[1.4, 0.9, 1.05], [0.5, 1.2, 1.5]]], dtype=torch.float64)
                deltas = torch.tensor([[[0.2, -0.1, 0.3], [-0.4, 0.1, 0.2]], [[0.3, -0.2, 0.1], [-0.1, 0.4, -0.3]]], dtype=ratios.dtype)
                old = torch.full_like(ratios, -2.)
                policy = old + ratios.log()
                ref = policy + deltas
                rewards = torch.tensor([[1., 3.], [12., 8.]], dtype=ratios.dtype)
                mask = torch.tensor([[[1, 1, 0], [1, 1, 1]], [[1, 0, 1], [0, 1, 1]]], dtype=torch.bool)
                # For these two-response groups, normalized advantages are [-1, 1] and [1, -1].
                advantages = [[-1., 1.], [1., -1.]]
                expected = 0.
                for b in range(2):
                    for g in range(2):
                        terms = []
                        for t in range(3):
                            if bool(mask[b, g, t]):
                                ratio = float(ratios[b, g, t])
                                delta = float(deltas[b, g, t])
                                advantage = advantages[b][g]
                                surrogate = min(ratio * advantage, min(max(ratio, 0.85), 1.15) * advantage)
                                terms.append(-surrogate + 0.17 * (math.expm1(delta) - delta))
                        expected += sum(terms) / len(terms) / 4
                loss = {fn}(policy, old, ref, rewards, mask, beta=0.17, epsilon=0.15)
                assert torch.allclose(loss, torch.tensor(expected, dtype=policy.dtype), atol=1e-12, rtol=1e-12), 'Match clipped token losses, the KL sign, and per-response averaging'
            """),
        },
        {
            "name": "Group-relative population normalization",
            "code": dedent("""
                import math
                import torch
                ratios = torch.tensor([[[0.8], [1.], [1.1]], [[0.8], [1.], [1.1]]], dtype=torch.float64)
                old = torch.full_like(ratios, -2.)
                policy = old + ratios.log()
                rewards = torch.tensor([[0., 1., 2.], [100., 102., 104.]], dtype=policy.dtype)
                mask = torch.ones_like(policy, dtype=torch.bool)
                loss = {fn}(policy, old, policy, rewards, mask, beta=0., epsilon=0.5)
                expected = torch.tensor(-math.sqrt(1.5) * 0.1, dtype=policy.dtype)
                assert torch.allclose(loss, expected, atol=1e-12, rtol=1e-12), 'Use each prompt group independently and population std, not sample or global std'
                shifted = rewards + torch.tensor([[7.], [-50.]], dtype=rewards.dtype)
                scaled = rewards * torch.tensor([[3.], [0.5]], dtype=rewards.dtype)
                assert torch.allclose({fn}(policy, old, policy, shifted, mask, beta=0., epsilon=0.5), loss), 'Group reward offsets must cancel'
                assert torch.allclose({fn}(policy, old, policy, scaled, mask, beta=0., epsilon=0.5), loss), 'Positive within-group reward scales must cancel away from the std floor'
            """),
        },
        {
            "name": "Positive and negative clipping branches and gradients",
            "code": dedent("""
                import torch
                ratios = torch.tensor([[[0.5, 0.9, 1.1, 1.5], [0.5, 0.9, 1.1, 1.5]]], dtype=torch.float64)
                old = torch.full_like(ratios, -2.)
                policy = (old + ratios.log()).requires_grad_()
                rewards = torch.tensor([[0., 2.]], dtype=policy.dtype)
                loss = {fn}(policy, old, old, rewards, torch.ones_like(policy, dtype=torch.bool), beta=0., epsilon=0.2)
                assert torch.allclose(loss, torch.tensor(0.075, dtype=policy.dtype), atol=1e-12, rtol=1e-12), 'Use the elementwise minimum before negation for both advantage signs'
                loss.backward()
                expected_grad = torch.tensor([[[0., 0.9, 1.1, 1.5], [-0.5, -0.9, -1.1, 0.]]], dtype=policy.dtype) / 8
                assert torch.allclose(policy.grad, expected_grad, atol=1e-12, rtol=1e-12), 'Clipping stops only positive-advantage ratios above the upper bound and negative-advantage ratios below the lower bound'
            """),
        },
        {
            "name": "Reference KL direction, beta, and near-zero accuracy",
            "code": dedent("""
                import math
                import torch
                policy = torch.full((1, 1, 2), -2., dtype=torch.float64, requires_grad=True)
                deltas = torch.tensor([[[0.2, -0.7]]], dtype=policy.dtype)
                ref = policy.detach() + deltas
                rewards = torch.tensor([[5.]], dtype=policy.dtype)
                mask = torch.ones_like(policy, dtype=torch.bool)
                loss = {fn}(policy, policy.detach(), ref, rewards, mask, beta=0.3)
                expected = 0.3 * sum(math.expm1(d) - d for d in (0.2, -0.7)) / 2
                assert torch.allclose(loss, torch.tensor(expected, dtype=policy.dtype), atol=1e-12, rtol=1e-12), 'Use ref minus current in the sampled-token KL penalty'
                loss.backward()
                expected_grad = 0.3 * (1 - deltas.exp()) / 2
                assert torch.allclose(policy.grad, expected_grad, atol=1e-12, rtol=1e-12), 'The reference penalty must remain differentiable through the current policy'
                assert {fn}(policy, policy.detach(), ref, rewards, mask, beta=0.).item() == 0., 'beta=0 disables the KL term'
                current = torch.tensor([[[-1.]]], dtype=torch.float64)
                reference = current + 1e-8
                delta = float((reference - current).item())
                tiny = {fn}(current, current, reference, rewards, torch.ones_like(current), beta=1.)
                expected_tiny = torch.tensor(delta * delta / 2 + delta ** 3 / 6, dtype=current.dtype)
                assert torch.allclose(tiny, expected_tiny, atol=1e-24, rtol=1e-6), 'Avoid cancellation in exp(delta) - delta - 1 near zero; expm1 is useful'
            """),
        },
        {
            "name": "Equal response weighting and empty-response semantics",
            "code": dedent("""
                import torch
                policy = torch.full((1, 2, 3), -1., dtype=torch.float64, requires_grad=True)
                rewards = torch.tensor([[0., 2.]], dtype=policy.dtype)
                mask = torch.tensor([[[1, 1, 1], [1, 0, 0]]], dtype=torch.bool)
                loss = {fn}(policy, policy.detach(), policy.detach(), rewards, mask, beta=0.)
                assert torch.allclose(loss, torch.tensor(0., dtype=policy.dtype), atol=1e-12), 'Average tokens inside each response first; unequal lengths must not change response weights'
                empty_response = torch.tensor([[[1, 1, 1], [0, 0, 0]]], dtype=torch.bool)
                loss = {fn}(policy, policy.detach(), policy.detach(), rewards, empty_response, beta=0.)
                assert torch.allclose(loss, torch.tensor(0.5, dtype=policy.dtype)), 'A fully masked response contributes zero but remains in the final B*G mean'
                all_empty = {fn}(policy, policy.detach(), policy.detach(), rewards, torch.zeros_like(mask), beta=0.04)
                assert torch.isfinite(all_empty) and all_empty.item() == 0., 'An all-masked batch has finite zero loss'
                all_empty.backward()
                assert torch.equal(policy.grad, torch.zeros_like(policy)), 'An all-masked batch has zero policy gradients'
            """),
        },
        {
            "name": "Gradients flow only through valid policy tokens",
            "code": dedent("""
                import math
                import torch
                policy = torch.tensor([[[-0.9, -1.1], [-0.8, -1.2]]], dtype=torch.float64, requires_grad=True)
                old = torch.full_like(policy, -1., requires_grad=True)
                deltas = torch.tensor([[[0.2, -0.3], [0.1, -0.2]]], dtype=policy.dtype)
                ref = (policy.detach() + deltas).requires_grad_()
                rewards = torch.tensor([[0., 2.]], dtype=policy.dtype, requires_grad=True)
                mask = torch.tensor([[[1., 0.], [1., 1.]]], dtype=policy.dtype, requires_grad=True)
                loss = {fn}(policy, old, ref, rewards, mask, beta=0.2, epsilon=0.5)
                loss.backward()
                expected = torch.tensor([[[(math.exp(0.1) + 0.2 * (1 - math.exp(0.2))) / 2, 0.], [(-math.exp(0.2) + 0.2 * (1 - math.exp(0.1))) / 4, (-math.exp(-0.2) + 0.2 * (1 - math.exp(-0.2))) / 4]]], dtype=policy.dtype)
                assert policy.grad is not None and torch.allclose(policy.grad, expected, atol=1e-12, rtol=1e-12), 'Respect masked per-response denominators and differentiate only the current policy'
                assert old.grad is None and ref.grad is None and rewards.grad is None and mask.grad is None, 'Rollout logps, reference logps, rewards, and token masks are fixed inputs'
            """),
        },
        {
            "name": "Nonfinite padding cannot poison the loss or gradients",
            "code": dedent("""
                import math
                import torch
                policy = torch.tensor([[[-1., float('nan'), float('inf')], [-1.1, float('-inf'), float('nan')]]], dtype=torch.float64, requires_grad=True)
                old = torch.tensor([[[-1.2, float('inf'), float('nan')], [-1.3, float('nan'), float('-inf')]]], dtype=policy.dtype)
                ref = torch.tensor([[[-0.9, float('-inf'), float('nan')], [-1., float('inf'), float('nan')]]], dtype=policy.dtype)
                mask = torch.tensor([[[1, 0, 0], [1, 0, 0]]], dtype=torch.bool)
                rewards = torch.tensor([[0., 2.]], dtype=policy.dtype)
                loss = {fn}(policy, old, ref, rewards, mask)
                expected = (math.exp(0.2) - 1.2) / 2 + 0.04 * (math.expm1(0.1) - 0.1)
                assert torch.isfinite(loss) and torch.allclose(loss, torch.tensor(expected, dtype=policy.dtype), atol=1e-12, rtol=1e-12), 'Sanitize padding before exponentiating; multiplication by a zero mask does not remove NaN/Inf'
                loss.backward()
                assert torch.isfinite(policy.grad).all(), 'Masked NaN/Inf padding must not create nonfinite gradients'
                assert torch.equal(policy.grad[~mask], torch.zeros_like(policy.grad[~mask])), 'Padding gradients must be zero'
            """),
        },
        {
            "name": "Constant rewards and singleton groups remain finite",
            "code": dedent("""
                import math
                import torch
                for shape in ((1, 1, 1), (2, 3, 2)):
                    policy = torch.full(shape, -1., dtype=torch.float64, requires_grad=True)
                    old = torch.full_like(policy, -0.8)
                    ref = torch.full_like(policy, -0.6)
                    rewards = torch.full(shape[:2], 7., dtype=policy.dtype)
                    loss = {fn}(policy, old, ref, rewards, torch.ones_like(policy, dtype=torch.bool), beta=0.2)
                    expected = torch.tensor(0.2 * (math.expm1(0.4) - 0.4), dtype=policy.dtype)
                    assert torch.isfinite(loss) and torch.allclose(loss, expected, atol=1e-12, rtol=1e-12), 'A singleton or constant-reward group has zero advantages; the KL term may remain'
                    loss.backward()
                    assert torch.isfinite(policy.grad).all(), 'Zero reward variance must not cause NaN gradients'
            """),
        },
        {
            "name": "Small group standard deviations use a floor",
            "code": dedent("""
                import torch
                old = torch.full((1, 2, 1), -2., dtype=torch.float64)
                ratios = torch.tensor([[[1.], [1.1]]], dtype=old.dtype)
                policy = old + ratios.log()
                rewards = torch.tensor([[0., 1e-8]], dtype=old.dtype)
                loss = {fn}(policy, old, policy, rewards, torch.ones_like(policy), beta=0.)
                expected = torch.tensor(-0.00025, dtype=old.dtype)
                assert torch.allclose(loss, expected, atol=1e-12, rtol=1e-12), 'Clamp population std to at least 1e-6 instead of dividing by tiny std or adding epsilon to every std'
            """),
        },
    ],
}
