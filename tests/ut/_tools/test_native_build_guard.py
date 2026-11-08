"""Regression tests for repeated and interrupted native wheel builds."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
MODULE_PATH = ROOT / "tools" / "native_build_guard.py"
SPEC = importlib.util.spec_from_file_location("native_build_guard_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def test_consecutive_builds_recreate_mutated_generated_tree(tmp_path: Path) -> None:
    build_temp = tmp_path / "build" / "native"
    with guard.exclusive_clean_build_tree(build_temp):
        mutated = build_temp / "kernel.o"
        mutated.write_text("postprocessed-not-an-object", encoding="utf-8")

    with guard.exclusive_clean_build_tree(build_temp):
        assert not mutated.exists()
        assert build_temp.is_dir()


def test_interrupted_build_is_cleaned_on_retry(tmp_path: Path) -> None:
    build_temp = tmp_path / "build" / "native"
    with pytest.raises(RuntimeError, match="interrupted"), guard.exclusive_clean_build_tree(build_temp):
        (build_temp / "partial.o").write_bytes(b"partial")
        raise RuntimeError("interrupted preprocessing")

    with guard.exclusive_clean_build_tree(build_temp):
        assert list(build_temp.iterdir()) == []


def test_parallel_build_for_same_tree_is_rejected(tmp_path: Path) -> None:
    build_temp = tmp_path / "build" / "native"
    with (
        guard.exclusive_clean_build_tree(build_temp),
        pytest.raises(RuntimeError, match="another native wheel build"),
        guard.exclusive_clean_build_tree(build_temp),
    ):
        pass


def test_different_build_trees_do_not_share_a_lock(tmp_path: Path) -> None:
    first = tmp_path / "build" / "first"
    second = tmp_path / "build" / "second"
    with guard.exclusive_clean_build_tree(first), guard.exclusive_clean_build_tree(second):
        assert first.is_dir()
        assert second.is_dir()


def test_repository_root_and_symlink_are_never_deleted(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="refusing unsafe"), guard.exclusive_clean_build_tree(Path.cwd()):
        pass

    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="refusing unsafe"), guard.exclusive_clean_build_tree(linked):
        pass
    assert target.is_dir()


def test_aclnn_build_recreates_generated_tree_but_preserves_cache() -> None:
    script = (ROOT / "csrc" / "build_aclnn.sh").read_text(encoding="utf-8")

    assert "rm -rf -- build output build_out" in script
    assert "rm -rf -- build_cache" not in script
