"""Generate verbatim reference explanations with the requested fenced-code prompt.

Run with the repository's Python environment and an authenticated Codex CLI:
  .venv/bin/python scripts/generate_explanations.py --run-dir /tmp/torchcode-explanations

Answers and logs are staged outside the repository. Add --publish after reviewing
them to update explanations/ and its provenance manifest. ChatGPT's private
instructions are unavailable; this uses a minimal assistant instruction instead
of Codex's coding-agent writing defaults, with the user prompt unchanged.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 uses the compatible TOML backport.
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.solutions import parse_solution_code, solution_files

MODEL = "gpt-6.1-sol"
EFFORT = "xhigh"
VERBOSITY = "high"
SYSTEM_INSTRUCTIONS = "You are a helpful assistant."
PROMPT_TEMPLATE = "```\n<solution_code>\n```\n\nexplain it"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def make_prompt(code):
    return "```\n" + code.rstrip() + "\n```\n\nexplain it"


def generation_context(auth_store):
    """Fingerprint request settings, rather than inferring them from old logs."""
    return {
        "model": MODEL, "reasoning_effort": EFFORT, "verbosity": VERBOSITY,
        "system_instructions": SYSTEM_INSTRUCTIONS,
        "built_in_instructions_replaced": True, "user_config_loaded": False,
        "ephemeral": True, "skip_git_repo_check": True, "sandbox": "read-only",
        "project_doc_max_bytes": 0, "personality": "none",
        "features": {"shell_tool": False, "apps": False, "skip_host_skill_discovery": True},
        "web_search": "disabled", "auth_store": auth_store,
    }


def context_fingerprint(context):
    return digest(json.dumps(context, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8"))


def completion_receipt(context, prompt, answer):
    """Construct only after successful execution and model/effort verification."""
    return {
        "schema_version": 1, "status": "completed", "process_returncode": 0,
        "context": context, "context_sha256": context_fingerprint(context),
        "prompt_sha256": digest(prompt), "raw_response_sha256": digest(answer),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def verify_receipt(receipt_path, context, prompt, answer):
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected = completion_receipt(context, prompt, answer)
        if (any(type(receipt.get(key)) is not type(value) or receipt.get(key) != value
                for key, value in expected.items() if key != "completed_at")
                or receipt["context_sha256"] != context_fingerprint(receipt["context"])):
            raise ValueError("Receipt does not match the successful response and exact context")
        completed = datetime.fromisoformat(receipt["completed_at"].replace("Z", "+00:00"))
        if completed.utcoffset() != timezone.utc.utcoffset(completed):
            raise ValueError("Receipt completion time is not UTC")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise RuntimeError(f"{receipt_path.stem}: unproven or mismatched cached response; use a fresh run directory") from error


def verified_answer(response_path, log_path):
    answer = response_path.read_bytes()
    if not answer.decode("utf-8").strip():
        raise RuntimeError(f"{response_path.stem}: empty answer")
    log = log_path.read_text(encoding="utf-8")
    if f"model: {MODEL}\n" not in log or f"reasoning effort: {EFFORT}\n" not in log:
        raise RuntimeError(f"{response_path.stem}: requested model/effort not confirmed")
    return answer


def generate(task_id, path, run_dir, codex, auth_store, instructions, context):
    if context != generation_context(auth_store):
        raise RuntimeError("Generation context changed")
    if instructions.read_bytes() != SYSTEM_INSTRUCTIONS.encode("utf-8"):
        raise RuntimeError("Assistant instructions changed before generation")
    stem = path.stem.removesuffix("_solution")
    code = parse_solution_code(path)
    prompt = make_prompt(code).encode("utf-8")
    prompt_path = run_dir / "prompts" / f"{stem}.txt"
    response_path = run_dir / "responses" / f"{stem}.md"
    log_path = run_dir / "logs" / f"{stem}.log"
    receipt_path = run_dir / "receipts" / f"{stem}.json"
    previous_prompt = prompt_path.read_bytes() if prompt_path.exists() else None
    started = time.monotonic()
    if response_path.exists() and response_path.read_bytes().strip():
        if previous_prompt != prompt:
            raise RuntimeError(f"{stem}: cached response does not match this prompt; use a fresh run directory")
        answer = verified_answer(response_path, log_path)
        verify_receipt(receipt_path, context, prompt, answer)
    else:
        prompt_path.write_bytes(prompt)
        receipt_path.unlink(missing_ok=True)
        command = [
            codex, "exec", "--ephemeral", "--ignore-user-config",
            "--skip-git-repo-check", "--sandbox", "read-only",
            "-c", f"cli_auth_credentials_store={json.dumps(auth_store)}",
            "--model", MODEL, "-c", f'model_reasoning_effort="{EFFORT}"',
            "-c", f'model_verbosity="{VERBOSITY}"',
            "-c", f"model_instructions_file={json.dumps(str(instructions))}",
            "-c", "project_doc_max_bytes=0", "-c", 'personality="none"',
            "-c", "features.shell_tool=false", "-c", "features.apps=false",
            "-c", "features.skip_host_skill_discovery=true",
            "-c", 'web_search="disabled"',
            "--cd", str(run_dir / "workspace"),
            "--output-last-message", str(response_path), "-",
        ]
        with log_path.open("wb") as log:
            result = subprocess.run(command, input=prompt, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{stem}: generation failed; see {log_path}")
        answer = verified_answer(response_path, log_path)
        if instructions.read_bytes() != SYSTEM_INSTRUCTIONS.encode("utf-8"):
            raise RuntimeError("Assistant instructions changed during generation")
        temporary = receipt_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(completion_receipt(context, prompt, answer), indent=2) + "\n", encoding="utf-8")
        temporary.replace(receipt_path)
    record = {
        "solution_file": path.relative_to(ROOT).as_posix(),
        "explanation_file": f"explanations/{stem}.md",
        "solution_code_sha256": digest(code.encode("utf-8")),
        "prompt_sha256": digest(prompt),
        "raw_response_sha256": digest(answer),
        "explanation_sha256": digest(answer),
    }
    return record, len(answer.decode("utf-8").split()), time.monotonic() - started


def publish_run(records, run_dir, context, manifest_bytes):
    """Revalidate every current source and receipt before publishing any file."""
    if context != generation_context(context["auth_store"]):
        raise RuntimeError("Generation context changed before publishing")
    paths = {path.relative_to(ROOT).as_posix(): path for path in solution_files(ROOT).values()}
    if len(records) != len(paths) or {record["solution_file"] for record in records} != paths.keys():
        raise RuntimeError("Publishing requires exactly one answer for every current solution")
    checked_answers = []
    for record in records:
        path = paths[record["solution_file"]]
        stem = path.stem.removesuffix("_solution")
        code = parse_solution_code(path)
        prompt = make_prompt(code).encode("utf-8")
        if (record["solution_code_sha256"] != digest(code.encode("utf-8"))
                or record["prompt_sha256"] != digest(prompt)
                or (run_dir / "prompts" / f"{stem}.txt").read_bytes() != prompt):
            raise RuntimeError(f"{stem}: source or prompt changed after generation")
        if record["explanation_file"] != f"explanations/{stem}.md":
            raise RuntimeError(f"{stem}: explanation path does not match its solution")
        answer = verified_answer(run_dir / "responses" / f"{stem}.md", run_dir / "logs" / f"{stem}.log")
        if digest(answer) != record["raw_response_sha256"] or record["explanation_sha256"] != digest(answer):
            raise RuntimeError(f"{stem}: staged answer changed after generation")
        verify_receipt(run_dir / "receipts" / f"{stem}.json", context, prompt, answer)
        checked_answers.append((ROOT / record["explanation_file"], answer))
    for path, answer in checked_answers:
        path.write_bytes(answer)
    (ROOT / "explanations/generation.json").write_bytes(manifest_bytes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--tasks", nargs="+", help="Generate a subset for preview")
    parser.add_argument("--publish", action="store_true", help="Publish a complete, reviewed run")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        parser.error("Use between 1 and 4 workers")
    run_dir = args.run_dir.resolve()
    if run_dir == ROOT or ROOT in run_dir.parents:
        parser.error("Stage generation outside the repository")
    codex = shutil.which("codex")
    if not codex:
        parser.error("Install and authenticate the Codex CLI first")
    # Keep the credential-store selection while omitting writing preferences.
    # No credentials or other user configuration are copied into the prompt.
    config_path = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "config.toml"
    auth_store = "file"
    if config_path.exists():
        auth_store = tomllib.loads(config_path.read_text(encoding="utf-8")).get("cli_auth_credentials_store", "file")
    if auth_store not in {"file", "keyring", "auto", "ephemeral"}:
        parser.error("Unrecognized Codex credential store")
    sources = solution_files(ROOT)
    if args.tasks:
        unknown = set(args.tasks) - sources.keys()
        if unknown:
            parser.error(f"Unknown tasks: {sorted(unknown)}")
        sources = {key: path for key, path in sources.items() if key in args.tasks}
    for folder in ("prompts", "responses", "logs", "workspace", "receipts"):
        (run_dir / folder).mkdir(parents=True, exist_ok=True)
    instructions = run_dir / "assistant-instructions.txt"
    instructions.write_text(SYSTEM_INSTRUCTIONS, encoding="utf-8")
    context = generation_context(auth_store)

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(generate, task_id, path, run_dir, codex, auth_store, instructions, context)
                   for task_id, path in sources.items()]
        for future in as_completed(futures):
            record, words, duration = future.result()
            records.append(record)
            print(f"{len(records):02}/{len(sources)} {record['explanation_file']}: {words} words ({duration:.0f}s)", flush=True)
    manifest = {
        "schema_version": 2,
        "model": MODEL,
        "reasoning_effort": EFFORT,
        "verbosity": VERBOSITY,
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_construction": '"```\\n" + parse_solution_code(notebook).rstrip() + "\\n```\\n\\nexplain it"',
        "system_instructions": SYSTEM_INSTRUCTIONS,
        "built_in_instructions_replaced": True,
        "user_config_loaded": False,
        "generation_method": "One fresh ephemeral Codex CLI session per solution, with an isolated workspace and explicit model/effort confirmed in each log.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "presentation_note": "Explanation files preserve the raw response bytes verbatim. Math compatibility and supplementary interactive examples are handled only by the viewer.",
        "files": sorted(records, key=lambda record: record["solution_file"]),
    }
    manifest_path = run_dir / "generation.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.publish:
        if args.tasks:
            parser.error("Publishing requires a complete run, without --tasks")
        publish_run(records, run_dir, context, manifest_path.read_bytes())
        print(f"Published {len(records)} verbatim explanations.", flush=True)


if __name__ == "__main__":
    main()
