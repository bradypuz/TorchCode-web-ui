"""Count added/replaced reference code without executing notebook cells.

Only marked implementation cells participate. The count is physical source
lines on the solution side of the diff, excluding imports, comments, docstrings,
and placeholders. AST-normalized statements preserve supplied scaffold even
when its whitespace or line wrapping changes.
"""

import ast
from copy import copy
from difflib import SequenceMatcher
import io
import json
from pathlib import Path
import re
import tokenize

from api.parser import TASK_ID_ALIASES
from api.solutions import ROOT, SOLUTION_MARKER, solution_files

TEMPLATE_MARKER = "# ✏️ YOUR IMPLEMENTATION HERE"
_BODY_FIELDS = {"body", "orelse", "finalbody"}
_IGNORED_TOKENS = {
    tokenize.ENCODING, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT,
    tokenize.NEWLINE, tokenize.NL, tokenize.COMMENT,
}
_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _marked_source(path: Path, marker: str) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise ValueError("Invalid notebook cells")
    sources = []
    for cell in notebook["cells"]:
        if not isinstance(cell, dict):
            raise ValueError("Invalid notebook cell")
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if isinstance(source, list) and all(isinstance(line, str) for line in source):
            source = "".join(source)
        if not isinstance(source, str):
            raise ValueError("Invalid notebook source")
        if marker in source:
            sources.append(source)
    if not sources:
        raise ValueError("Notebook has no marked implementation cell")
    return "\n\n".join(sources)


def _is_placeholder(node: ast.stmt) -> bool:
    return isinstance(node, ast.Pass) or (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and node.value.value is Ellipsis
    )


def _is_docstring(node: ast.stmt) -> bool:
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str))


def _child_blocks(node: ast.AST, path: tuple = ()):
    """Include try handlers/match cases as well as ordinary statement bodies."""
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            if field in _BODY_FIELDS and all(isinstance(item, ast.stmt) for item in value):
                yield path + (field,), value
            else:
                for index, item in enumerate(value):
                    if isinstance(item, ast.AST) and not isinstance(item, ast.stmt):
                        yield from _child_blocks(item, path + (field, index))
        elif isinstance(value, ast.AST) and not isinstance(value, ast.stmt):
            yield from _child_blocks(value, path + (field,))


def _header_key(node: ast.stmt) -> str:
    header = copy(node)
    for field in _BODY_FIELDS | {"handlers", "cases"}:
        if hasattr(header, field):
            setattr(header, field, [])
    return ast.dump(header, include_attributes=False)


def _code_units(source: str) -> list[tuple[tuple, set[int]]]:
    """Pair normalized statement/header keys with their physical source lines."""
    tree = ast.parse(source)
    source_lines = source.splitlines()
    tokens = [token for token in tokenize.generate_tokens(io.StringIO(source).readline)
              if token.type not in _IGNORED_TOKENS]
    units = []

    def position(node, end=False):
        line = node.end_lineno if end else node.lineno
        byte_column = node.end_col_offset if end else node.col_offset
        # AST columns are UTF-8 byte offsets; tokenize uses character offsets.
        column = len(source_lines[line - 1].encode("utf-8")[:byte_column].decode("utf-8"))
        return line, column

    def span(node):
        start, end = position(node), position(node, end=True)
        if isinstance(node, _DEFINITIONS) and node.decorator_list:
            decorator_start = position(node.decorator_list[0])
            start = (decorator_start[0], max(0, decorator_start[1] - 1))
        return start, end

    def visit_body(body, scope=(), docstring_scope=False):
        for index, node in enumerate(body):
            if (isinstance(node, (ast.Import, ast.ImportFrom)) or _is_placeholder(node)
                    or (docstring_scope and index == 0 and _is_docstring(node))):
                continue
            blocks = list(_child_blocks(node))
            children = sorted(span(child) for _, block in blocks for child in block)
            start, end = span(node)
            # Separate the main header from continuation headers such as else:
            # and except:. Child statements are compared independently below.
            parts = {}
            for token in tokens:
                if token.start < start or token.end > end:
                    continue
                if any(child_start <= token.start and token.end <= child_end
                       for child_start, child_end in children):
                    continue
                part = sum(child_end <= token.start for _, child_end in children)
                parts.setdefault(part, []).append(token)
            for index, (_, part_tokens) in enumerate(sorted(parts.items())):
                key = (_header_key(node) if index == 0 else
                       tuple((token.type, token.string) for token in part_tokens))
                lines = {line for token in part_tokens
                         for line in range(token.start[0], token.end[0] + 1)
                         if source_lines[line - 1].strip()}
                units.append((scope + (key,), lines))
            context = (type(node).__name__, node.name) if isinstance(node, _DEFINITIONS) else (type(node).__name__,)
            for path, block in blocks:
                visit_body(block, scope + (context, path),
                           docstring_scope=isinstance(node, _DEFINITIONS) and path == ("body",))

    visit_body(tree.body, docstring_scope=True)
    return units


def count_implementation_lines(template_source: str, solution_source: str) -> int:
    """Count solution lines added/replaced, rather than net growth in code size."""
    template_units = _code_units(template_source)
    solution_units = _code_units(solution_source)
    matcher = SequenceMatcher(a=[key for key, _ in template_units],
                              b=[key for key, _ in solution_units], autojunk=False)
    added_lines = set()
    for operation, _, _, start, end in matcher.get_opcodes():
        if operation in {"insert", "replace"}:
            for _, lines in solution_units[start:end]:
                added_lines.update(lines)
    return len(added_lines)


def get_implementation_line_counts(root: Path = ROOT) -> dict[str, int | None]:
    """Unknown or unreadable notebook pairs remain null, never a fabricated zero."""
    templates = {}
    for path in sorted((root / "templates").glob("*.ipynb")):
        match = re.fullmatch(r"\d+_(.+)\.ipynb", path.name)
        if match and match.group(1) != "welcome":
            stem = match.group(1)
            templates[TASK_ID_ALIASES.get(stem, stem)] = path
    solutions = solution_files(root)
    counts = {}
    for task_id in templates.keys() | solutions.keys():
        counts[task_id] = None
        template, solution = templates.get(task_id), solutions.get(task_id)
        if template is None or solution is None:
            continue
        try:
            counts[task_id] = count_implementation_lines(
                _marked_source(template, TEMPLATE_MARKER),
                _marked_source(solution, SOLUTION_MARKER),
            )
        except (OSError, ValueError, SyntaxError, tokenize.TokenError):
            continue
    return counts
