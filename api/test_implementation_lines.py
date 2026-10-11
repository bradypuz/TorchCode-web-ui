"""Coverage for the Journey's reference implementation line counts."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from api import main
from api.implementation_lines import count_implementation_lines, get_implementation_line_counts
from torch_judge.tasks import TASKS


# Audited against the checked-in implementation cells, including physical
# continuation lines and added helper definitions, but excluding scaffold.
EXPECTED_COUNTS = {
    "relu": 1, "softmax": 3, "linear": 4, "layernorm": 4, "attention": 4,
    "mha": 16, "batchnorm": 12, "rmsnorm": 2, "causal_attention": 7,
    "gqa": 19, "sliding_window": 8, "linear_attention": 7, "gpt2_block": 27,
    "kv_cache": 27, "mlp": 4, "cross_entropy": 2, "dropout": 5,
    "embedding": 2, "gelu": 1, "weight_init": 5, "gradient_clipping": 7,
    "conv2d": 11, "cross_attention": 15, "rope": 12, "flash_attention": 21,
    "lora": 7, "vit_patch": 9, "moe": 22, "adam": 20, "cosine_lr": 6,
    "gradient_accumulation": 9, "topk_sampling": 13, "beam_search": 19,
    "speculative_decoding": 17, "bpe": 40, "int8_quantization": 10,
    "dpo_loss": 3, "grpo_loss": 10, "ppo_loss": 6, "linear_regression": 30,
    "opd_loss": 31,
}


def write_notebook(path, cells):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cells": [{"cell_type": "code", "source": cell}
                                          for cell in cells]}), encoding="utf-8")


class ImplementationLineTests(unittest.TestCase):
    def test_every_problem_matches_the_audited_count_including_mha_alias(self):
        counts = get_implementation_line_counts()
        self.assertEqual(set(counts), set(TASKS))
        self.assertEqual(counts, EXPECTED_COUNTS)

    def test_replacements_count_even_when_the_number_of_source_lines_is_equal(self):
        template = "def f(x):\n    value = x + 1\n    return value\n"
        solution = "def f(x):\n    value = x * 2\n    return value\n"
        self.assertEqual(count_implementation_lines(template, solution), 1)
        self.assertEqual(count_implementation_lines(template, "def f(x):\n    return x\n"), 1)

    def test_supplied_scaffold_is_unchanged_despite_formatting_and_docstring_edits(self):
        template = '''class Model:
    """Provided description."""
    def __init__(self, size=1):
        super().__init__()
        self.size = size
    def forward(self, x):
        pass
'''
        solution = '''class Model:
    """Updated description.
    With a second line.
    """
    def __init__(
        self, size = 1,
    ):
        super(
        ).__init__()  # supplied initialization
        self.size=(size)
    def forward(self, x):
        return x * self.size  # new implementation
'''
        self.assertEqual(count_implementation_lines(template, solution), 1)

    def test_changed_condition_does_not_count_the_preserved_body(self):
        template = "def f(x):\n    if x > 0:\n        return x\n    return 0\n"
        solution = "def f(x):\n    if x >= 0:\n        return x\n    return 0\n"
        self.assertEqual(count_implementation_lines(template, solution), 1)

    def test_new_helpers_and_multiline_code_count_physical_lines(self):
        template = "def f(x):\n    ...  # placeholder\n"
        solution = '''def f(x):
    def helper(y):
        return (
            y * 2
        )
    return helper(x)
'''
        self.assertEqual(count_implementation_lines(template, solution), 5)

    def test_repeated_scaffold_in_a_different_method_still_counts(self):
        template = '''class Encoder:
    def __init__(self):
        self.merges = []
    def train(self):
        pass
'''
        solution = '''class Encoder:
    def __init__(self):
        self.merges = []
    def train(self):
        self.merges = []
'''
        self.assertEqual(count_implementation_lines(template, solution), 1)

    def test_multiple_changed_statements_on_one_physical_line_count_once(self):
        self.assertEqual(count_implementation_lines("a = 1; b = 2\n", "a = 3; b = 4\n"), 1)

    def test_only_marked_cells_count_excluding_imports_comments_and_docstrings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_notebook(root / "templates/01_relu.ipynb", [
                "print('template setup')",
                ["# ✏️ YOUR IMPLEMENTATION HERE\n", "def relu(x):\n    pass\n"],
            ])
            write_notebook(root / "solutions/01_relu_solution.ipynb", [
                "print('solution setup'); raise RuntimeError('must not execute')",
                '''# ✅ SOLUTION
import torch
from math import (
    sqrt,
)
def relu(x):
    """Reference documentation.
    This is not an implementation line.
    """
    # Explain the expression below.
    return x  # inline comment
''',
                "print('demo'); check('relu')",
                "# ✅ SOLUTION\ndef helper(x):\n    return x\n",
            ])
            self.assertEqual(get_implementation_line_counts(root), {"relu": 3})

    def test_missing_or_malformed_pairs_are_unknown_and_never_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(get_implementation_line_counts(root), {})
            template = root / "templates/01_relu.ipynb"
            solution = root / "solutions/01_relu_solution.ipynb"
            write_notebook(template, ["# ✏️ YOUR IMPLEMENTATION HERE\ndef relu(x):\n    pass\n"])
            self.assertEqual(get_implementation_line_counts(root), {"relu": None})
            solution.parent.mkdir()
            for content in ("not json", "[]", '{"cells": {}}',
                            '{"cells": [{"cell_type": "code", "source": 1}]}'):
                with self.subTest(content=content):
                    solution.write_text(content)
                    self.assertEqual(get_implementation_line_counts(root), {"relu": None})
            for source in ("def relu(x):\n    return x\n", "# ✅ SOLUTION\ndef relu(:\n"):
                with self.subTest(source=source):
                    write_notebook(solution, [source])
                    self.assertEqual(get_implementation_line_counts(root), {"relu": None})
            write_notebook(solution, ["# ✅ SOLUTION\ndef relu(x):\n    return x\n"])
            template.write_text("{bad notebook")
            self.assertEqual(get_implementation_line_counts(root), {"relu": None})

    def test_valid_identical_implementation_has_a_real_zero_count(self):
        source = "def f(x):\n    return x\n"
        self.assertEqual(count_implementation_lines(source, source), 0)

    def test_task_list_preserves_metadata_and_exposes_only_the_nullable_count(self):
        with patch.object(main, "IMPLEMENTATION_LINES", {"relu": 1, "dropout": 5}):
            rows = main.list_tasks()
        self.assertEqual(len(rows), len(TASKS))
        for row in rows:
            with self.subTest(task=row["id"]):
                self.assertEqual(set(row), {"id", "title", "difficulty", "implementation_lines"})
                self.assertEqual(row["title"], TASKS[row["id"]]["title"])
                self.assertEqual(row["difficulty"], TASKS[row["id"]].get("difficulty", "Unknown"))
                expected = {"relu": 1, "dropout": 5}.get(row["id"])
                self.assertEqual(row["implementation_lines"], expected)


if __name__ == "__main__":
    unittest.main()
