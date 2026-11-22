"""INT8 Quantized Linear task."""

TASK = {
    "title": "INT8 Quantized Linear",
    "difficulty": "Hard",
    "function_name": "Int8Linear",
    "hint": "Per-output-channel scale = (weight.abs().amax(dim=1, keepdim=True) / 127).clamp(min=1e-10). Quantize: round(weight/scale).clamp(-127,127).to(int8). Store the same clamped scale for dequantization. Forward: dequantize, matmul, and add optional bias.",
    "tests": [
        {
            "name": "Weight is int8",
            "code": "\nimport torch, torch.nn as nn\nw = torch.randn(32, 16)\nq = {fn}(w)\nassert isinstance(q, nn.Module)\nassert q.weight_int8.dtype == torch.int8, f'dtype: {q.weight_int8.dtype}'\n"
        },
        {
            "name": "Values in [-128, 127]",
            "code": "\nimport torch\nq = {fn}(torch.randn(64, 32) * 10)\nassert q.weight_int8.min() >= -128 and q.weight_int8.max() <= 127\n"
        },
        {
            "name": "Dequantized close to original",
            "code": "\nimport torch\ntorch.manual_seed(0)\nw = torch.randn(16, 8)\nq = {fn}(w)\nw_recon = q.weight_int8.float() * q.scale\nassert (w - w_recon).abs().max() < 0.1, 'Quantization error too large'\n"
        },
        {
            "name": "Forward output shape",
            "code": "\nimport torch\nq = {fn}(torch.randn(8, 4), torch.randn(8))\nout = q(torch.randn(2, 4))\nassert out.shape == (2, 8), f'Shape: {out.shape}'\n"
        },
        {
            "name": "Weight is buffer not parameter",
            "code": "\nimport torch\nq = {fn}(torch.randn(4, 4))\nparam_names = [n for n, _ in q.named_parameters()]\nassert 'weight_int8' not in param_names, 'weight_int8 should be a buffer'\nassert 'scale' not in param_names, 'scale should be a buffer'\n"
        },
        {
            "name": "Zero row has a safe scale",
            "code": "\nimport torch\nw = torch.tensor([[0., 0., 0.], [1., -2., 3.]])\nq = {fn}(w)\nassert torch.isfinite(q.scale).all() and (q.scale > 0).all(), 'Scales must be finite and positive, including for zero rows'\nassert (q.weight_int8[0] == 0).all(), 'Zero row must quantize to zeros'\nout = q(torch.tensor([[1., 2., 3.], [-1., 0., 2.]]))\nassert torch.isfinite(out).all(), 'Zero rows must not produce non-finite output'\nassert (out[:, 0] == 0).all(), 'Zero row must produce zero output without bias'\n"
        },
        {
            "name": "Tiny and ordinary rows preserve scale and bias",
            "code": "\nimport torch\nw = torch.tensor([[1e-8, -5e-9], [1., -0.5]])\nbias = torch.tensor([3e-9, 0.25])\nq = {fn}(w, bias)\nw_recon = q.weight_int8.float() * q.scale\nallowed_weight_error = torch.tensor([[1e-10], [1. / 127]])\nassert ((w_recon - w).abs() <= allowed_weight_error).all(), 'Use the same effective scale for quantization and reconstruction, including tiny rows'\nx = torch.tensor([[1., 2.], [-2., 0.5]])\nout = q(x)\nexpected = x @ w.T + bias\nassert torch.isfinite(out).all(), 'Output must be finite'\nassert ((out - expected).abs() <= torch.tensor([3e-10, 0.02])).all(), 'Quantized outputs must remain close to the linear transform with bias'\n"
        }
    ]
}
