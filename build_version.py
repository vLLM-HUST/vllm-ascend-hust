"""Fail-closed source version resolution for reproducible HUST builds."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

from packaging.version import InvalidVersion, Version
from setuptools_scm import get_version

MINIMUM_TRUSTED_SOURCE_VERSION = Version("0.23")


def _git_output(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "A complete Git checkout with synchronized upstream tags is required to build vLLM Ascend."
        ) from exc


def validate_source_version(version: str) -> str:
    """Reject fallback or stale versions before a wheel can be produced."""

    try:
        parsed = Version(version)
    except InvalidVersion as exc:
        raise RuntimeError(f"Invalid vLLM Ascend source version: {version!r}") from exc
    if parsed < MINIMUM_TRUSTED_SOURCE_VERSION:
        raise RuntimeError(
            "Refusing stale or fallback vLLM Ascend source version "
            f"{version!r}; expected >= {MINIMUM_TRUSTED_SOURCE_VERSION}. "
            "Fetch canonical upstream tags and use a trusted main-snapshot tag."
        )
    return version


def resolve_trusted_scm_version(
    root: str | Path,
    describe_command: Sequence[str],
    *,
    write_to: str | None = None,
) -> str:
    """Resolve a traceable version only from complete, tagged Git history."""

    checkout = Path(root).resolve()
    if _git_output(checkout, "rev-parse", "--is-shallow-repository") == "true":
        raise RuntimeError(
            "Refusing to build vLLM Ascend from a shallow checkout; "
            "fetch complete history and canonical upstream tags first."
        )

    # Run the exact describe command first so a missing reachable tag cannot be
    # hidden by setuptools-scm's archive or fallback heuristics.
    _git_output(checkout, *describe_command[1:])
    try:
        version_options = dict(
            root=checkout,
            git_describe_command=list(describe_command),
        )
        if write_to is not None:
            version_options["write_to"] = write_to
        resolved = get_version(**version_options)
    except LookupError as exc:
        raise RuntimeError("Unable to derive vLLM Ascend version from tagged Git history.") from exc
    return validate_source_version(resolved)
