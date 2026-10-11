"""Generation provenance tests; subprocess calls are mocked throughout."""

import builtins
import json
import os
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import generate_explanations as generation


class ExplanationGenerationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.root = self.directory / "repo"
        self.run_dir = self.directory / "run"
        for folder in ("prompts", "responses", "logs", "workspace", "receipts"):
            (self.run_dir / folder).mkdir(parents=True)
        (self.root / "explanations").mkdir(parents=True)
        self.instructions = self.run_dir / "assistant-instructions.txt"
        self.instructions.write_text(generation.SYSTEM_INSTRUCTIONS, encoding="utf-8")
        self.path = self.write_source("01_relu", "def relu(x):\n    return x\n")
        self.context = generation.generation_context("keyring")
        self.answer = b"\nThis relu implementation returns its input.  \n"
        root_patch = patch.object(generation, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        run_patch = patch.object(generation.subprocess, "run", side_effect=self.successful_process)
        self.process = run_patch.start()
        self.addCleanup(run_patch.stop)

    def write_source(self, stem, source):
        path = self.root / "solutions" / f"{stem}_solution.ipynb"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"cells": [{"cell_type": "code", "source": "# ✅ SOLUTION\n" + source}]}))
        return path

    def successful_process(self, command, *, input, stdout, stderr):
        response = Path(command[command.index("--output-last-message") + 1])
        response.write_bytes(self.answer)
        stdout.write(f"model: {generation.MODEL}\nreasoning effort: {generation.EFFORT}\n".encode())
        return SimpleNamespace(returncode=0)

    def generate(self, path=None, context=None):
        return generation.generate("relu", path or self.path, self.run_dir, "mock-codex",
                                   "keyring", self.instructions, context or self.context)[0]

    def receipt_path(self, stem="01_relu"):
        return self.run_dir / "receipts" / f"{stem}.json"

    def set_published_sentinels(self, stems=("01_relu",)):
        for stem in stems:
            (self.root / "explanations" / f"{stem}.md").write_bytes(b"previous answer")
        (self.root / "explanations/generation.json").write_bytes(b"previous manifest")

    def assert_published_unchanged(self, stems=("01_relu",)):
        for stem in stems:
            self.assertEqual((self.root / "explanations" / f"{stem}.md").read_bytes(), b"previous answer")
        self.assertEqual((self.root / "explanations/generation.json").read_bytes(), b"previous manifest")

    def test_python_310_loads_with_the_compatible_tomli_parser(self):
        # Exercise module loading as on Python 3.10 without needing another
        # interpreter. Both parsers expose the same loads API; supply the
        # installed parser for the backport import in this isolated test.
        original_import = builtins.__import__
        imports = []

        def import_without_stdlib_toml(name, *args, **kwargs):
            if name == "tomllib":
                raise ModuleNotFoundError("No module named 'tomllib'", name="tomllib")
            if name == "tomli":
                imports.append(name)
                return generation.tomllib
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_stdlib_toml):
            module = runpy.run_path(generation.__file__, run_name="generation_compatibility_test")
        self.assertEqual(imports, ["tomli"])
        self.assertEqual(module["tomllib"].loads('cli_auth_credentials_store="keyring"\n'),
                         {"cli_auth_credentials_store": "keyring"})

    def test_success_writes_exact_receipt_and_verified_cache_avoids_another_process(self):
        record = self.generate()
        receipt = json.loads(self.receipt_path().read_text())
        self.assertEqual(receipt["process_returncode"], 0)
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["context"], self.context)
        self.assertEqual(receipt["context_sha256"], generation.context_fingerprint(self.context))
        self.assertEqual(receipt["prompt_sha256"], record["prompt_sha256"])
        self.assertEqual(receipt["raw_response_sha256"], generation.digest(self.answer))
        self.assertEqual(self.generate(), record)
        self.process.assert_called_once()
        command = self.process.call_args.args[0]
        self.assertIn('cli_auth_credentials_store="keyring"', command)
        self.assertIn('model_verbosity="high"', command)
        self.assertIn("--ignore-user-config", command)

    def test_changed_base_instructions_or_verbosity_cannot_relabel_a_cached_answer(self):
        self.generate()
        for setting, value in (("SYSTEM_INSTRUCTIONS", "Explain with a different assistant."),
                               ("VERBOSITY", "low")):
            with self.subTest(setting=setting), patch.object(generation, setting, value):
                self.instructions.write_text(generation.SYSTEM_INSTRUCTIONS, encoding="utf-8")
                context = generation.generation_context("keyring")
                self.assertNotEqual(generation.context_fingerprint(context),
                                    generation.context_fingerprint(self.context))
                with self.assertRaisesRegex(RuntimeError, "unproven or mismatched"):
                    self.generate(context=context)
        self.instructions.write_text(generation.SYSTEM_INSTRUCTIONS, encoding="utf-8")
        self.process.assert_called_once()

    def test_changed_prompt_or_answer_cannot_reuse_a_receipt(self):
        self.generate()
        self.write_source("01_relu", "def relu(x):\n    return x * 2\n")
        with self.assertRaisesRegex(RuntimeError, "does not match this prompt"):
            self.generate()
        self.write_source("01_relu", "def relu(x):\n    return x\n")
        (self.run_dir / "responses/01_relu.md").write_bytes(b"Edited explanation")
        with self.assertRaisesRegex(RuntimeError, "unproven or mismatched"):
            self.generate()
        self.process.assert_called_once()

    def test_nonzero_exit_with_nonempty_output_has_no_success_receipt_or_cache(self):
        def failed_process(*args, **kwargs):
            self.successful_process(*args, **kwargs)
            return SimpleNamespace(returncode=1)
        self.process.side_effect = failed_process
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            self.generate()
        self.assertTrue((self.run_dir / "responses/01_relu.md").read_bytes())
        self.assertFalse(self.receipt_path().exists())
        with self.assertRaisesRegex(RuntimeError, "unproven or mismatched"):
            self.generate()
        self.process.assert_called_once()

    def test_empty_or_unconfirmed_successful_process_gets_no_receipt(self):
        self.answer = b" \n"
        with self.assertRaisesRegex(RuntimeError, "empty answer"):
            self.generate()
        self.assertFalse(self.receipt_path().exists())
        self.answer = b"An answer"
        def unconfirmed_process(command, **kwargs):
            result = self.successful_process(command, **kwargs)
            kwargs["stdout"].seek(0)
            kwargs["stdout"].truncate()
            kwargs["stdout"].write(b"model: other-model\n")
            return result
        self.process.side_effect = unconfirmed_process
        with self.assertRaisesRegex(RuntimeError, "model/effort not confirmed"):
            self.generate()
        self.assertFalse(self.receipt_path().exists())

    def test_instructions_changed_during_execution_get_no_receipt(self):
        def changed_process(*args, **kwargs):
            result = self.successful_process(*args, **kwargs)
            self.instructions.write_text("changed during execution")
            return result
        self.process.side_effect = changed_process
        with self.assertRaisesRegex(RuntimeError, "instructions changed during"):
            self.generate()
        self.assertFalse(self.receipt_path().exists())

    def test_invalid_completion_receipt_is_rejected(self):
        self.generate()
        valid = json.loads(self.receipt_path().read_text())
        invalid = [None, "not json", {}, {**valid, "process_returncode": 1},
                   {**valid, "context_sha256": "wrong"}, {**valid, "completed_at": "2026-10-10"}]
        for receipt in invalid:
            with self.subTest(receipt=receipt):
                self.receipt_path().unlink(missing_ok=True)
                if receipt is not None:
                    self.receipt_path().write_text(receipt if isinstance(receipt, str) else json.dumps(receipt))
                with self.assertRaisesRegex(RuntimeError, "unproven or mismatched"):
                    self.generate()
        self.process.assert_called_once()

    def test_publish_preserves_verified_response_bytes(self):
        record = self.generate()
        self.set_published_sentinels()
        manifest = b'{"validated": true}\n'
        generation.publish_run([record], self.run_dir, self.context, manifest)
        self.assertEqual((self.root / record["explanation_file"]).read_bytes(), self.answer)
        self.assertEqual((self.root / "explanations/generation.json").read_bytes(), manifest)

    def test_publish_rechecks_all_sources_before_writing_any_answers(self):
        second = self.write_source("02_softmax", "def my_softmax(x):\n    return x\n")
        records = [self.generate(), self.generate(second)]
        self.set_published_sentinels(("01_relu", "02_softmax"))
        self.write_source("02_softmax", "def my_softmax(x):\n    return x * 2\n")
        with self.assertRaisesRegex(RuntimeError, "source or prompt changed"):
            generation.publish_run(records, self.run_dir, self.context, b"new manifest")
        self.assert_published_unchanged(("01_relu", "02_softmax"))

    def test_publish_rechecks_prompts_answers_and_success_receipts(self):
        record = self.generate()
        self.set_published_sentinels()
        cases = [(self.run_dir / "prompts/01_relu.txt", b"changed prompt", "source or prompt changed"),
                 (self.run_dir / "responses/01_relu.md", b"changed answer", "staged answer changed"),
                 (self.receipt_path(), b"{}", "unproven or mismatched")]
        for path, contents, message in cases:
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(contents)
                with self.assertRaisesRegex(RuntimeError, message):
                    generation.publish_run([record], self.run_dir, self.context, b"new manifest")
                self.assert_published_unchanged()
                path.write_bytes(original)

    def test_publish_rejects_a_new_solution_without_an_answer(self):
        record = self.generate()
        self.set_published_sentinels()
        self.write_source("02_softmax", "def my_softmax(x):\n    return x\n")
        with self.assertRaisesRegex(RuntimeError, "every current solution"):
            generation.publish_run([record], self.run_dir, self.context, b"new manifest")
        self.assert_published_unchanged()

    def test_main_cannot_publish_unproven_cache_even_with_matching_prompt_and_log(self):
        self.generate()
        self.receipt_path().unlink()
        self.process.reset_mock()
        self.set_published_sentinels()
        client_home = self.directory / "client"
        client_home.mkdir()
        (client_home / "config.toml").write_text('cli_auth_credentials_store="keyring"\n')
        argv = ["generate_explanations.py", "--run-dir", str(self.run_dir), "--workers", "1", "--publish"]
        with patch.object(generation.shutil, "which", return_value="mock-codex"), \
                patch.object(generation.sys, "argv", argv), patch.dict(os.environ, {"CODEX_HOME": str(client_home)}):
            with self.assertRaisesRegex(RuntimeError, "unproven or mismatched"):
                generation.main()
        self.process.assert_not_called()
        self.assert_published_unchanged()


if __name__ == "__main__":
    unittest.main()
