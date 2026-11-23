"""REINFORCE policy-gradient loss task."""

TASK = {
    "title": "REINFORCE Loss",
    "difficulty": "Medium",
    "function_name": "reinforce_loss",
    "hint": (
        "Apply log_softmax across actions, then select logps[row, actions[row]] "
        "using torch.arange on the logits' device (or gather). "
        "Return the negative batch mean of rewards.detach() times those log-probabilities."
    ),
    "tests": [
        {
            "name": "Scalar loss preserves dtype and device",
            "code": "\n"
            "import torch\n"
            "logits = torch.tensor([[2., 1., 0.], [0., 1., 2.]], dtype=torch.float64, requires_grad=True)\n"
            "actions = torch.tensor([0, 1])\n"
            "rewards = torch.tensor([1., 0.], dtype=torch.float64)\n"
            "loss = {fn}(logits, actions, rewards)\n"
            "assert isinstance(loss, torch.Tensor) and loss.dim() == 0, 'Loss must be a scalar Tensor'\n"
            "assert loss.dtype == logits.dtype and loss.device == logits.device, 'Preserve the logits dtype and device'\n"
            "assert loss.requires_grad, 'Keep the loss connected to the policy logits'\n"
            "assert torch.allclose(loss, torch.tensor(0.2038029822, dtype=logits.dtype)), 'Worked example should give approximately 0.203803'\n"
        },
        {
            "name": "Chosen actions and signed rewards",
            "code": "\n"
            "import torch\n"
            "import torch.nn.functional as F\n"
            "logits = torch.tensor([[2., -1., 0.], [0.5, 2., -0.5], [-2., 0., 3.], [1., -2., 0.]], dtype=torch.float64)\n"
            "actions = torch.tensor([2, 0, 1, 1])\n"
            "rewards = torch.tensor([1.5, -0.5, 0., 2.], dtype=logits.dtype)\n"
            "expected = (F.cross_entropy(logits, actions, reduction='none') * rewards).mean()\n"
            "loss = {fn}(logits, actions, rewards)\n"
            "assert torch.allclose(loss, expected, atol=1e-9, rtol=1e-9), 'Select one action per row, retain reward signs, and average over the batch'\n"
        },
        {
            "name": "Policy gradient and detached rewards",
            "code": "\n"
            "import torch\n"
            "import torch.nn.functional as F\n"
            "logits = torch.tensor([[2., 1., -1.], [-1., 0., 2.], [0., 3., 1.]], dtype=torch.float64, requires_grad=True)\n"
            "actions = torch.tensor([0, 1, 2])\n"
            "rewards = torch.tensor([2., -1., 0.], dtype=logits.dtype, requires_grad=True)\n"
            "loss = {fn}(logits, actions, rewards)\n"
            "loss.backward()\n"
            "expected = (logits.detach().softmax(dim=-1) - F.one_hot(actions, num_classes=3)) * rewards.detach()[:, None] / 3\n"
            "assert logits.grad is not None and torch.allclose(logits.grad, expected, atol=1e-9, rtol=1e-9), 'The logits gradient must be reward * (probability - one_hot(action)) / batch_size'\n"
            "assert rewards.grad is None, 'Treat rewards as fixed weights: gradients must not flow into rewards'\n"
        },
        {
            "name": "Zero rewards still count in the batch mean",
            "code": "\n"
            "import torch\n"
            "logits = torch.zeros(4, 2, dtype=torch.float64, requires_grad=True)\n"
            "actions = torch.tensor([0, 1, 0, 1])\n"
            "rewards = torch.tensor([2., 0., 0., 0.], dtype=logits.dtype)\n"
            "loss = {fn}(logits, actions, rewards)\n"
            "expected = torch.log(torch.tensor(2., dtype=logits.dtype)) / 2\n"
            "assert torch.allclose(loss, expected, atol=1e-9, rtol=1e-9), 'Divide by all four examples, not by the reward sum or nonzero reward count'\n"
            "zero_loss = {fn}(logits, actions, torch.zeros_like(rewards))\n"
            "assert torch.isfinite(zero_loss) and zero_loss.item() == 0., 'An all-zero reward batch must have a finite zero loss'\n"
        },
        {
            "name": "Numerically stable with large logits",
            "code": "\n"
            "import torch\n"
            "logits = torch.tensor([[1000., 1001., 999.], [-1000., 0., 1000.]], dtype=torch.float64, requires_grad=True)\n"
            "actions = torch.tensor([0, 0])\n"
            "rewards = torch.tensor([1., 0.25], dtype=logits.dtype)\n"
            "expected = (torch.logsumexp(torch.tensor([0., 1., -1.], dtype=logits.dtype), dim=0) + 500.) / 2\n"
            "loss = {fn}(logits, actions, rewards)\n"
            "assert torch.isfinite(loss) and torch.allclose(loss, expected, atol=1e-9, rtol=1e-9), 'Compute log probabilities stably even when softmax would underflow'\n"
            "loss.backward()\n"
            "assert logits.grad is not None and torch.isfinite(logits.grad).all(), 'Large finite logits must produce finite gradients'\n"
        },
        {
            "name": "Single example and single action",
            "code": "\n"
            "import torch\n"
            "logits = torch.tensor([[1., 3.]], dtype=torch.float64)\n"
            "loss = {fn}(logits, torch.tensor([0]), torch.tensor([-2.], dtype=logits.dtype))\n"
            "expected = -2 * torch.log1p(torch.exp(torch.tensor(2., dtype=logits.dtype)))\n"
            "assert loss.dim() == 0 and torch.allclose(loss, expected), 'Support a batch containing one example, including negative rewards'\n"
            "logits = torch.tensor([[4.], [-3.], [10.]], dtype=torch.float64, requires_grad=True)\n"
            "loss = {fn}(logits, torch.zeros(3, dtype=torch.long), torch.tensor([1., -2., 0.], dtype=logits.dtype))\n"
            "assert loss.dim() == 0 and loss.item() == 0., 'With one possible action its log probability is zero'\n"
            "loss.backward()\n"
            "assert logits.grad is not None and torch.equal(logits.grad, torch.zeros_like(logits)), 'A one-action policy has zero policy gradient'\n"
        },
    ],
}
