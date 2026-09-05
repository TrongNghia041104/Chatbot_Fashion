"""Audit structure and Python syntax of the split Research Demo v3 notebooks."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks" / "research_demo_v3_split"
REQUIRED_MARKDOWN_FIELDS = (
    "**Tác dụng chính:**",
    "**Đầu vào (Input):**",
    "**Đầu ra (Output):**",
)


@dataclass
class FunctionIssue:
    """Describe a function whose documentation does not meet the notebook standard."""

    cell_index: int
    function_name: str
    missing_sections: list[str]


def cell_source(cell: dict) -> str:
    """Return a notebook cell source as one string.

    Args:
        cell: Raw Jupyter cell dictionary.

    Returns:
        Concatenated source text.
    """
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def is_complex_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Decide whether a function needs a full Args/Returns docstring.

    Args:
        node: Function node parsed from a code cell.

    Returns:
        ``True`` for functions with parameters and non-trivial control flow or size.
    """
    line_count = (getattr(node, "end_lineno", node.lineno) or node.lineno) - node.lineno + 1
    control_nodes = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match)
    has_control_flow = any(isinstance(child, control_nodes) for child in ast.walk(node))
    parameter_count = len(node.args.posonlyargs) + len(node.args.args) + len(node.args.kwonlyargs)
    return line_count >= 8 or has_control_flow or parameter_count >= 3


def function_issues(tree: ast.Module, cell_index: int) -> list[FunctionIssue]:
    """Find complex functions missing standard docstring sections.

    Args:
        tree: Parsed module AST for one code cell.
        cell_index: Position of that cell in the notebook.

    Returns:
        Documentation issues found in the cell.
    """
    issues: list[FunctionIssue] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not is_complex_function(node):
            continue
        docstring = ast.get_docstring(node, clean=False) or ""
        missing = [section for section in ("Args:", "Returns:") if section not in docstring]
        if missing:
            issues.append(FunctionIssue(cell_index, node.name, missing))
    return issues


def has_mixed_definition_and_execution(tree: ast.Module) -> bool:
    """Detect cells that mix top-level definitions with runtime statements.

    Args:
        tree: Parsed module AST for one code cell.

    Returns:
        ``True`` when definitions and observable execution coexist.
    """
    has_definition = any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in tree.body)
    runtime_types = (ast.Expr, ast.For, ast.While, ast.Try, ast.With, ast.Match)
    has_runtime = any(isinstance(node, runtime_types) for node in tree.body)
    return has_definition and has_runtime


def has_section_contract(cells: list[dict], code_index: int) -> bool:
    """Check whether a code cell belongs to a documented business section.

    Args:
        cells: Notebook cells.
        code_index: Index of the code cell being checked.

    Returns:
        ``True`` when the nearest preceding Markdown has all required fields.
    """
    for index in range(code_index - 1, -1, -1):
        cell = cells[index]
        if cell.get("cell_type") == "markdown":
            source = cell_source(cell)
            return all(field in source for field in REQUIRED_MARKDOWN_FIELDS)
    return False


def audit_notebook(path: Path) -> dict:
    """Audit one notebook without modifying it.

    Args:
        path: Notebook path.

    Returns:
        Summary containing syntax, Markdown, docstring, and cell-mixing issues.
    """
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    report = {
        "syntax_errors": [],
        "markdown_missing": [],
        "mixed_cells": [],
        "function_issues": [],
        "code_with_outputs": 0,
        "collapsed_implementation": 0,
    }

    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            report["code_with_outputs"] += 1
        if cell.get("metadata", {}).get("jupyter", {}).get("source_hidden"):
            report["collapsed_implementation"] += 1
        source = cell_source(cell)
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            report["syntax_errors"].append((index, exc.lineno, exc.msg))
            continue

        report["function_issues"].extend(function_issues(tree, index))
        if has_mixed_definition_and_execution(tree):
            report["mixed_cells"].append(index)

        if not has_section_contract(cells, index):
            report["markdown_missing"].append(index)
    return report


def main() -> None:
    """Print one compact structural audit for every split notebook."""
    total_issues = 0
    for path in sorted(NOTEBOOK_DIR.rglob("*.ipynb")):
        report = audit_notebook(path)
        issue_count = sum(
            len(report[key])
            for key in ("syntax_errors", "markdown_missing", "mixed_cells", "function_issues")
        )
        total_issues += issue_count
        print(
            f"{path.relative_to(NOTEBOOK_DIR)}: issues={issue_count} | syntax={len(report['syntax_errors'])} | "
            f"markdown={len(report['markdown_missing'])} | mixed={len(report['mixed_cells'])} | "
            f"docstrings={len(report['function_issues'])} | outputs={report['code_with_outputs']} | "
            f"collapsed={report['collapsed_implementation']}"
        )
        if report["syntax_errors"]:
            print(f"  syntax: {report['syntax_errors']}")
        if report["mixed_cells"]:
            print(f"  mixed cells: {report['mixed_cells']}")
        if report["function_issues"]:
            details = [
                f"cell {issue.cell_index}:{issue.function_name}({','.join(issue.missing_sections)})"
                for issue in report["function_issues"]
            ]
            print("  docstrings: " + "; ".join(details))
    print(f"TOTAL_ISSUES={total_issues}")


if __name__ == "__main__":
    main()
