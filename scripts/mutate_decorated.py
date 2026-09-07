"""Exercise decorator-hidden bodies in a disposable copy, never in src/."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path


def expose_decorated_bodies(tree: ast.Module) -> dict[str, str]:
    """Keep descriptors/decorators, but give their bodies mutable names."""
    names: dict[str, str] = {}
    containers: list[tuple[str, ast.Module | ast.ClassDef]] = [("", tree)]
    containers.extend(
        (node.name, node)
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    )
    for prefix, container in containers:
        statements: list[ast.stmt] = []
        for node in container.body:
            statements.append(node)
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = node.decorator_list
            if not decorators:
                continue
            if all(
                isinstance(decorator, ast.Name)
                and decorator.id in {"classmethod", "staticmethod", "overload"}
                for decorator in decorators
            ):
                continue
            # Only the single-decorator forms currently used by localstub.
            # Fail rather than silently changing arbitrary decorator semantics.
            if len(decorators) != 1:
                raise ValueError(f"Unsupported decorators on {node.name}")
            decorator = decorators[0]
            original = node.name
            helper = f"mutation_body_{original}_{node.lineno}"
            names[f"{prefix + '.' if prefix else ''}{helper}"] = (
                f"{prefix + '.' if prefix else ''}{original}:{node.lineno}"
            )
            node.name = helper
            node.decorator_list = []
            # The original name remains bound until this assignment, so
            # @handler.setter still resolves the preceding getter descriptor.
            statements.append(
                ast.Assign(
                    targets=[ast.Name(id=original, ctx=ast.Store())],
                    value=ast.Call(
                        func=decorator,
                        args=[ast.Name(id=helper, ctx=ast.Load())],
                        keywords=[],
                    ),
                )
            )
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "lru_cache"
            ):
                # Mutmut forks after baseline tests warm this cache. Clear
                # inherited entries so each worker executes its mutated body.
                statements.extend(
                    ast.parse(
                        "import os as _mutmut_os\n"
                        "_mutmut_os.register_at_fork(\n"
                        f"    after_in_child={original}.cache_clear\n"
                        ")\n"
                    ).body
                )
        container.body = statements
    ast.fix_missing_locations(tree)
    return names


def prepare(root: Path, workspace: Path) -> None:
    workspace.mkdir(exist_ok=True)
    # Renamed helpers need fresh test-to-function tracking, not cached names
    # from an earlier transformed copy. This supplemental pass is small.
    cache = workspace / "mutants"
    if cache.exists():
        shutil.rmtree(cache)
    for directory in ("src", "tests", "scripts"):
        destination = workspace / directory
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(
            root / directory,
            destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    shutil.copy2(root / "pyproject.toml", workspace / "pyproject.toml")
    mapping: dict[str, dict[str, str]] = {}
    for path in sorted((workspace / "src").rglob("*.py")):
        tree = ast.parse(path.read_text())
        names = expose_decorated_bodies(tree)
        if names:
            mapping[str(path.relative_to(workspace))] = names
            path.write_text(ast.unparse(tree) + "\n")
    (workspace / "decorated-bodies.json").write_text(
        json.dumps(mapping, indent=2) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["run", "results", "show", "browse", "export-cicd-stats"],
    )
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    workspace = root / "mutants-decorated"
    extra = args.args
    if args.command == "run":
        prepare(root, workspace)
        extra = ["--max-children", "4", *(extra or ["*mutation_body_*"])]
    elif not workspace.exists():
        parser.error("Run the decorated mutation scan first")
    return subprocess.run(
        [sys.executable, "-m", "mutmut", args.command, *extra],
        cwd=workspace,
        check=False,
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
