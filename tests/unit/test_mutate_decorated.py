from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.mutate_decorated import expose_decorated_bodies, prepare


def test_exposed_properties_preserve_getters_and_setters() -> None:
    tree = ast.parse(
        """
class Example:
    current = 2

    @property
    def value(self):
        return self.current + 1

    @value.setter
    def value(self, value):
        self.current = value

    @classmethod
    def answer(cls):
        return 42

instance = Example()
before = instance.value
instance.value = 8
after = instance.value
answer = Example.answer()
"""
    )
    mapping = expose_decorated_bodies(tree)
    namespace: dict[str, Any] = {}

    exec(compile(tree, "<test>", "exec"), namespace)

    assert namespace["before"] == 3
    assert namespace["after"] == 9
    assert namespace["answer"] == 42
    assert len(mapping) == 2
    assert all(name.startswith("Example.value:") for name in mapping.values())


@pytest.mark.parametrize(
    "source",
    [
        """
from functools import lru_cache

@lru_cache(maxsize=2)
def cached(value):
    return value + 1

assert cached(1) == cached(1) == 2
assert cached.cache_info().hits == 1
""",
        """
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def context():
    yield 42

async def run():
    async with context() as value:
        assert value == 42

asyncio.run(run())
""",
    ],
)
def test_exposed_functions_preserve_decorator_behavior(source: str) -> None:
    tree = ast.parse(source)
    mapping = expose_decorated_bodies(tree)

    exec(compile(tree, "<test>", "exec"), {})

    assert len(mapping) == 1


def test_multiple_unsupported_decorators_fail_explicitly() -> None:
    tree = ast.parse("@first\n@second\ndef function(): return 1\n")

    with pytest.raises(ValueError, match="Unsupported decorators on function"):
        expose_decorated_bodies(tree)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="mutmut requires fork")
def test_exposed_cache_is_cleared_in_mutation_workers() -> None:
    tree = ast.parse(
        """
import os
import sys
from functools import lru_cache

@lru_cache(maxsize=2)
def cached(value):
    return value + 1

cached(1)
pid = os.fork()
if pid == 0:
    sys.exit(0 if cached.cache_info().currsize == 0 else 1)
_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0
assert cached.cache_info().currsize == 1
"""
    )
    expose_decorated_bodies(tree)

    result = subprocess.run(
        [sys.executable, "-c", ast.unparse(tree)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_prepare_isolates_source_and_refreshes_deleted_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("src", "tests", "scripts"):
        (root / name).mkdir()
    (root / "pyproject.toml").write_text("[tool.mutmut]\n")
    source = "class Example:\n    @property\n    def value(self): return 1\n"
    (root / "src" / "example.py").write_text(source)
    (root / "src" / "plain.py").write_text("VALUE = 1\n")
    workspace = tmp_path / "copy"

    prepare(root, workspace)
    (workspace / "src" / "stale.py").touch()
    (workspace / "mutants").mkdir()
    (workspace / "mutants" / "stale-stats.json").touch()
    prepare(root, workspace)

    assert (root / "src" / "example.py").read_text() == source
    assert (
        "mutation_body_value" in (workspace / "src" / "example.py").read_text()
    )
    assert not (workspace / "src" / "stale.py").exists()
    assert not (workspace / "mutants").exists()
    assert (workspace / "src" / "plain.py").read_text() == "VALUE = 1\n"
    mapping = json.loads((workspace / "decorated-bodies.json").read_text())
    assert list(mapping) == ["src/example.py"]
