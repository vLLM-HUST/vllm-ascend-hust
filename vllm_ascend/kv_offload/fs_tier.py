# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ascend filesystem secondary tier for multi-tier KV cache offloading.

This mirrors the upstream vLLM filesystem tier but disables O_DIRECT by default,
which avoids alignment-related EINVAL failures on 3FS/FUSE paths. Current vLLM
builds without the experimental tiering package can still import this module;
constructing the tier manager raises an ImportError with the missing dependency.
"""

import functools
import logging
import os
import random
import threading
from typing import Any

_TIERING_IMPORT_ERROR: ModuleNotFoundError | None
try:
    from vllm.v1.kv_offload.tiering.base import JobMetadata
    from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
except ModuleNotFoundError as exc:
    JobMetadata = Any
    FileSystemTierManager = object
    _TIERING_IMPORT_ERROR = exc
else:
    _TIERING_IMPORT_ERROR = None

logger = logging.getLogger(__name__)
_thread_local = threading.local()
_created_dirs: set[str] = set()
_created_dirs_lock = threading.Lock()


def _ensure_dir(dir_path: str) -> None:
    if dir_path in _created_dirs:
        return
    os.makedirs(dir_path, exist_ok=True)
    with _created_dirs_lock:
        _created_dirs.add(dir_path)


def _get_tmp_suffix() -> str:
    try:
        return _thread_local.tmp_suffix
    except AttributeError:
        _thread_local.tmp_suffix = f"_{random.randint(0, 2**63 - 1)}.tmp"
        return _thread_local.tmp_suffix


def store_block(
    dest_path: str,
    buffer: memoryview,
    offset: int,
    block_size: int,
    open_flags: int,
) -> None:
    """Write one KV block atomically using temp-file + replace."""
    if os.path.exists(dest_path):
        return

    dir_path = os.path.dirname(dest_path)
    tmp_path = dest_path + _get_tmp_suffix()
    _ensure_dir(dir_path)

    view_slice = buffer.cast("B")[offset : offset + block_size]
    try:
        try:
            fd = os.open(tmp_path, open_flags, 0o644)
        except FileNotFoundError:
            _created_dirs.discard(dir_path)
            os.makedirs(dir_path, exist_ok=True)
            with _created_dirs_lock:
                _created_dirs.add(dir_path)
            fd = os.open(tmp_path, open_flags, 0o644)
        try:
            total = len(view_slice)
            written = 0
            while written < total:
                n = os.write(fd, view_slice[written:])
                if n <= 0:
                    raise OSError(f"Short write: expected {total} bytes, wrote {written}")
                written += n
        finally:
            os.close(fd)
        os.replace(tmp_path, dest_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError as cleanup_exc:
            logger.warning("Failed to remove temp file %s: %s", tmp_path, cleanup_exc)
        raise


def load_block(
    source_path: str,
    view: memoryview,
    offset: int,
    block_size: int,
    open_flags: int,
) -> None:
    """Read one KV block into view and remove corrupt/short entries."""
    fd: int | None = None
    view_slice = view.cast("B")[offset : offset + block_size]
    try:
        fd = os.open(source_path, open_flags)
        read_total = 0
        while read_total < block_size:
            n = os.readv(fd, [view_slice[read_total:]])
            if n == 0:
                break
            read_total += n
        if read_total < block_size:
            raise OSError(f"Short read: expected {block_size} bytes, read {read_total}")
    except Exception:
        try:
            os.remove(source_path)
        except OSError as cleanup_exc:
            logger.warning("Failed to remove unreadable file %s: %s", source_path, cleanup_exc)
        raise
    finally:
        if fd is not None:
            os.close(fd)


class AscendFileSystemTierManager(FileSystemTierManager):
    """fs_python secondary tier without requiring O_DIRECT."""

    def __init__(
        self,
        offloading_spec,
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        use_direct_io: bool = False,
    ):
        if _TIERING_IMPORT_ERROR is not None:
            raise ImportError(
                "AscendFileSystemTierManager requires vllm.v1.kv_offload.tiering"
            ) from _TIERING_IMPORT_ERROR

        direct = getattr(os, "O_DIRECT", 0) if use_direct_io else 0
        if use_direct_io and direct == 0:
            logger.warning("use_direct_io=True but O_DIRECT is unavailable; using buffered I/O.")
        self._store_flags: int = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_TRUNC | direct
        self._load_flags: int = os.O_RDONLY | direct

        super().__init__(
            offloading_spec=offloading_spec,
            primary_kv_view=primary_kv_view,
            tier_type=tier_type,
            root_dir=root_dir,
            n_read_threads=n_read_threads,
            n_write_threads=n_write_threads,
        )

    def submit_store(self, job_metadata: JobMetadata) -> None:
        tasks = (
            functools.partial(
                store_block,
                self.file_mapper.get_file_name(key),
                self._primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
                self._store_flags,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._pool.enqueue_store(job_metadata.job_id, len(job_metadata.keys), tasks)

    def submit_load(self, job_metadata: JobMetadata) -> None:
        tasks = (
            functools.partial(
                load_block,
                self.file_mapper.get_file_name(key),
                self._primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
                self._load_flags,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._pool.enqueue_load(job_metadata.job_id, len(job_metadata.keys), tasks)
