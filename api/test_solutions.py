"""Coverage for the notebook-to-reference reader and explanation inventory."""

import ast
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

from fastapi import HTTPException

from api.solutions import ROOT, get_solution, parse_solution_code, solution_files
from torch_judge.tasks import TASKS


class SolutionTests(unittest.TestCase):
    def test_every_problem_has_its_own_solution_and_explanation(self):
        self.assertEqual(set(solution_files()), set(TASKS))
        explanations = []
        for task_id, task in TASKS.items():
            with self.subTest(task_id=task_id):
                result = get_solution(task_id)
                self.assertEqual(result["id"], task_id)
                self.assertEqual(result["title"], task["title"])
                code = result["code"]
                tree = ast.parse(code)
                definitions = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
                self.assertIn(task["function_name"], definitions)
                self.assertNotIn("google.colab", code)
                self.assertNotIn("get_ipython", code)
                self.assertNotIn("from torch_judge import check", code)
                self.assertNotIn("# Demo", code)
                explanation = result["explanation"]
                explanations.append(explanation)
                self.assertTrue(explanation.strip())
                explanation_path = ROOT / result["explanation_file"]
                self.assertEqual(explanation, explanation_path.read_bytes().decode("utf-8"))
                # Allow the model's own headings and length while checking that
                # each answer discusses the supplied problem's topic or symbol.
                generic_words = {"implement", "simple", "layer", "with", "loss", "block"}
                topics = {word.casefold() for word in re.findall(r"[A-Za-z][A-Za-z0-9]*", task["title"])
                          if len(word) >= 3 and word.casefold() not in generic_words}
                topics.add(task["function_name"].casefold())
                normalization_topics = {
                    "batchnorm": {"batch normalization"},
                    "layernorm": {"layer normalization"},
                    "rmsnorm": {"rms normalization", "root mean square normalization"},
                }
                topics.update(normalization_topics.get(task_id, set()))
                normalized = re.sub(r"\s+", " ", explanation.replace("\\_", "_").casefold())
                self.assertTrue(any(re.search(r"\b" + re.escape(topic) + r"\b", normalized)
                                    for topic in topics), "Explanation does not reference this problem")
        self.assertEqual(len(set(explanations)), len(TASKS))

    def test_mha_uses_the_multihead_notebook_alias(self):
        self.assertEqual(get_solution("mha")["solution_file"], "solutions/06_multihead_attention_solution.ipynb")

    def test_generation_provenance_matches_every_exact_fenced_prompt_and_saved_answer(self):
        manifest = json.loads((ROOT / "explanations/generation.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["model"], "gpt-6.1-sol")
        self.assertEqual(manifest["reasoning_effort"], "xhigh")
        self.assertEqual(manifest["verbosity"], "high")
        self.assertEqual(manifest["prompt_template"], "```\n<solution_code>\n```\n\nexplain it")
        self.assertEqual(manifest["system_instructions"], "You are a helpful assistant.")
        self.assertIs(manifest["built_in_instructions_replaced"], True)
        self.assertIs(manifest["user_config_loaded"], False)
        generated_at = datetime.fromisoformat(manifest["generated_at"].replace("Z", "+00:00"))
        self.assertIsNotNone(generated_at.tzinfo)
        self.assertEqual(generated_at.utcoffset(), timedelta(0))
        self.assertIn("verbatim", manifest["presentation_note"].casefold())

        paths = sorted(solution_files().values())
        records = manifest["files"]
        self.assertEqual([record["solution_file"] for record in records],
                         [path.relative_to(ROOT).as_posix() for path in paths])
        for path, record in zip(paths, records):
            with self.subTest(solution=record["solution_file"]):
                code = parse_solution_code(path)
                prompt = "```\n" + code.rstrip() + "\n```\n\nexplain it"
                explanation_path = ROOT / "explanations" / (path.stem.removesuffix("_solution") + ".md")
                answer = explanation_path.read_bytes()
                self.assertTrue(answer.decode("utf-8").strip())
                self.assertEqual(record["explanation_file"], explanation_path.relative_to(ROOT).as_posix())
                self.assertEqual(record["solution_code_sha256"], hashlib.sha256(code.encode("utf-8")).hexdigest())
                self.assertEqual(record["prompt_sha256"], hashlib.sha256(prompt.encode("utf-8")).hexdigest())
                self.assertEqual(record["explanation_sha256"], hashlib.sha256(answer).hexdigest())
                transforms = record.get("presentation_transforms", [])
                self.assertIn(transforms, ([], ["remove_imports"], ["remove_imports", "renumber_sections"]))
                if transforms:
                    self.assertNotEqual(record["explanation_sha256"], record["raw_response_sha256"])
                    self.assertNotRegex(answer.decode("utf-8"), r"(?m)^\s*(?:import\s|from\s+\S+\s+import\s)")
                else:
                    self.assertEqual(record["explanation_sha256"], record["raw_response_sha256"])

    def test_parser_keeps_imports_and_multiple_solution_cells_only(self):
        cells = [
            "try:\n import google.colab\n get_ipython()\nexcept ImportError:\n pass",
            "import torch\nfrom math import (\n sqrt,\n pi,\n)\nprint('setup')",
            "# ✅ SOLUTION\ndef first(x):\n return x + 1",
            "print('demo')",
            "# ✅ SOLUTION\ndef second(x):\n return first(x)",
            "from torch_judge import check\ncheck('relu')",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "solution.ipynb"
            path.write_text(json.dumps({"cells": [{"cell_type": "code", "source": source} for source in cells]}))
            code = parse_solution_code(path)
            self.assertIn("from math import (", code)
            self.assertIn("def first", code)
            self.assertIn("def second", code)
            self.assertNotIn("print(", code)
            self.assertNotIn("torch_judge", code)

    def test_unknown_task_and_path_traversal_do_not_read_files(self):
        for task_id in ("missing", "../../README.md"):
            with self.subTest(task_id=task_id), self.assertRaises(HTTPException) as caught:
                get_solution(task_id)
            self.assertEqual(caught.exception.status_code, 404)

    def test_missing_explanation_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "solutions").mkdir()
            source = solution_files()["relu"].read_text()
            (root / "solutions/01_relu_solution.ipynb").write_text(source)
            with self.assertRaises(HTTPException) as caught:
                get_solution("relu", root)
            self.assertEqual(caught.exception.status_code, 503)

    def test_explanation_is_returned_verbatim_without_an_added_heading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "solutions").mkdir()
            (root / "explanations").mkdir()
            solution_path = root / "solutions/01_relu_solution.ipynb"
            solution_path.write_bytes(solution_files()["relu"].read_bytes())
            explanation = "\n\nThis `relu` function keeps positive values.  \r\n\r\n```python\r\nrelu(x)\r\n```\r\n"
            (root / "explanations/01_relu.md").write_bytes(explanation.encode("utf-8"))
            result = get_solution("relu", root)
            self.assertEqual(result["explanation"], explanation)
            self.assertEqual(result["code"], parse_solution_code(solution_path))
            self.assertEqual(set(result), {"id", "title", "difficulty", "code", "explanation",
                                           "solution_file", "explanation_file"})

    def test_whitespace_only_explanation_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "solutions").mkdir()
            (root / "explanations").mkdir()
            (root / "solutions/01_relu_solution.ipynb").write_bytes(solution_files()["relu"].read_bytes())
            explanation_path = root / "explanations/01_relu.md"
            for explanation in ("", " \n\t\r\n"):
                with self.subTest(explanation=explanation):
                    explanation_path.write_bytes(explanation.encode("utf-8"))
                    with self.assertRaises(HTTPException) as caught:
                        get_solution("relu", root)
                    self.assertEqual(caught.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
