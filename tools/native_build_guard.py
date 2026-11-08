"""Exclusive, recoverable lifecycle for the generated native build tree."""

from __future__ import annotations

import fcntl
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def exclusive_clean_build_tree(build_temp: str | os.PathLike[str]) -> Iterator[Path]:
    """Lock and recreate a generated tree before Ascend object preprocessing.

    AscendC preprocessing may mutate compiler-produced objects in place. Reusing
    that tree can feed an already-processed file back to the linker. A sibling
    lock rejects concurrent wheel builds, and recreating the tree makes both a
    completed prior build and an interrupted prior build recoverable.
    """
    requested_path = Path(build_temp)
    build_path = requested_path.resolve()
    protected_paths = {Path.cwd().resolve(), Path.home().resolve(), Path("/")}
    if requested_path.is_symlink() or build_path in protected_paths or len(build_path.parts) < 3:
        raise RuntimeError(f"refusing unsafe generated build tree {build_path}")
    build_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = build_path.with_name(f".{build_path.name}.vllm-ascend.lock")
    with lock_path.open("a+b") as lock_file:
        os.chmod(lock_path, 0o600)
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another native wheel build owns generated tree {build_path}") from exc
        try:
            if build_path.exists():
                shutil.rmtree(build_path)
            build_path.mkdir(parents=True)
            yield build_path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
