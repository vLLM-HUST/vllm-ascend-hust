#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import builtins
import importlib
import importlib.util
import sys
import types
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.ut.base import PytestBase

_MISSING = object()


def load_camem_module(acl_module=_MISSING):
    fake_c = types.ModuleType("vllm_ascend.vllm_ascend_C")
    fake_c.__spec__ = importlib.util.spec_from_loader("vllm_ascend.vllm_ascend_C", loader=None)
    fake_c.init_module = MagicMock()  # type: ignore[attr-defined]
    fake_c.python_create_and_map = MagicMock()  # type: ignore[attr-defined]
    fake_c.python_unmap_and_release = MagicMock()  # type: ignore[attr-defined]

    module_entries = {"vllm_ascend.vllm_ascend_C": fake_c}
    if acl_module is _MISSING:
        fake_acl = types.ModuleType("acl")
        fake_acl.__spec__ = importlib.util.spec_from_loader("acl", loader=None)
        fake_rt = types.ModuleType("acl.rt")
        fake_rt.__spec__ = importlib.util.spec_from_loader("acl.rt", loader=None)
        fake_rt.memcpy = MagicMock()  # type: ignore[attr-defined]
        fake_acl.rt = fake_rt  # type: ignore[attr-defined]
        module_entries.update({"acl": fake_acl, "acl.rt": fake_rt})
    elif acl_module is not None:
        module_entries.update({"acl": acl_module, "acl.rt": acl_module.rt})

    with patch.dict(sys.modules, module_entries, clear=False):
        return importlib.reload(importlib.import_module("vllm_ascend.device_allocator.camem"))


def dummy_malloc(args):
    pass


def dummy_free(ptr):
    return (0, 0, 0, 0)


class TestCaMem(PytestBase):
    def test_camem_loads_memcpy_from_top_level_acl_module(self):
        fake_acl = types.ModuleType("acl")
        fake_rt = types.ModuleType("acl.rt")
        fake_rt.memcpy = object()  # type: ignore[attr-defined]
        fake_acl.rt = fake_rt  # type: ignore[attr-defined]

        module = cast(Any, load_camem_module(fake_acl))

        assert module.memcpy is cast(Any, fake_rt).memcpy

    def test_camem_discovers_acl_site_packages_when_sys_path_is_missing(self, tmp_path, monkeypatch):
        site_packages = tmp_path / "Ascend" / "cann" / "python" / "site-packages"
        site_packages.mkdir(parents=True)
        (site_packages / "acl.py").write_text(
            "class rt:\n    memcpy = object()\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path / "Ascend" / "cann"))
        monkeypatch.setattr(
            sys,
            "path",
            [path for path in sys.path if "Ascend" not in path and "cann" not in path.lower()],
            raising=False,
        )

        with patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("acl", None)
            sys.modules.pop("acl.rt", None)
            module = load_camem_module(None)

        assert module.memcpy is not None

    def test_camem_disables_sleep_mode_for_broken_acl_install(self):
        original_import = builtins.__import__

        def broken_acl_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "acl":
                raise RuntimeError("broken acl")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=broken_acl_import):
            module = load_camem_module(None)

        assert module.memcpy is None

    @pytest.mark.parametrize(
        "handle",
        [
            (1, 2, 3),
            ("device", 99),
            (None,),
        ],
    )
    def test_create_and_map_calls_python_create_and_map(self, handle):
        camem = load_camem_module()
        with patch.object(camem, "python_create_and_map") as mock_create:
            camem.create_and_map(handle)
            mock_create.assert_called_once_with(*handle)

    @pytest.mark.parametrize(
        "handle",
        [
            (42, "bar"),
            ("foo",),
        ],
    )
    def test_unmap_and_release_calls_python_unmap_and_release(self, handle):
        camem = load_camem_module()
        with patch.object(camem, "python_unmap_and_release") as mock_release:
            camem.unmap_and_release(handle)
            mock_release.assert_called_once_with(*handle)

    def test_get_pluggable_allocator(self):
        camem = load_camem_module()
        mock_allocator_instance = MagicMock()
        mock_allocator_class = MagicMock()
        mock_init_module = MagicMock()
        with (
            patch.object(camem.torch.npu.memory, "NPUPluggableAllocator", mock_allocator_class),
            patch.object(camem, "init_module", mock_init_module),
        ):
            mock_allocator_class.return_value = mock_allocator_instance

            def side_effect_malloc_and_free(malloc_fn, free_fn):
                malloc_fn((1, 2, 3))
                free_fn(123)

            mock_init_module.side_effect = side_effect_malloc_and_free

            allocator = camem.get_pluggable_allocator(dummy_malloc, dummy_free)
            mock_init_module.assert_called_once_with(dummy_malloc, dummy_free)
            assert allocator == mock_allocator_instance

    def test_singleton_behavior(self):
        camem = load_camem_module()
        instance1 = camem.CaMemAllocator.get_instance()
        instance2 = camem.CaMemAllocator.get_instance()
        assert instance1 is instance2

    def test_python_malloc_and_free_callback(self):
        camem = load_camem_module()
        allocator = camem.CaMemAllocator.get_instance()

        # mock allocation_handle
        handle = (1, 100, 1234, 0)
        allocator.current_tag = "test_tag"

        allocator.python_malloc_callback(handle)
        # check pointer_to_data store data
        ptr = handle[2]
        assert ptr in allocator.pointer_to_data
        data = allocator.pointer_to_data[ptr]
        assert data.handle == handle
        assert data.tag == "test_tag"

        # check free callback with cpu_backup_tensor
        data.cpu_backup_tensor = torch.zeros(1)
        result_handle = allocator.python_free_callback(ptr)
        assert result_handle == handle
        assert ptr not in allocator.pointer_to_data
        assert data.cpu_backup_tensor is None

    def test_sleep_offload_and_discard(self):
        camem = load_camem_module()
        allocator = camem.CaMemAllocator.get_instance()
        mock_memcpy = MagicMock()
        mock_unmap = MagicMock()

        # prepare allocation， one tag match，one not match
        handle1 = (1, 10, 1000, 0)
        data1 = camem.AllocationData(handle1, "tag1")
        handle2 = (2, 20, 2000, 0)
        data2 = camem.AllocationData(handle2, "tag2")
        allocator.pointer_to_data = {
            1000: data1,
            2000: data2,
        }

        # Mock torch.empty to force pin_memory=False
        original_torch_empty = torch.empty

        def mock_torch_empty(*args, **kwargs):
            # If pin_memory was explicitly set to True, change it to False
            if "pin_memory" in kwargs and kwargs["pin_memory"] is True:
                kwargs["pin_memory"] = False
            return original_torch_empty(*args, **kwargs)

        with (
            patch.object(camem, "unmap_and_release", mock_unmap),
            patch.object(camem, "memcpy", mock_memcpy),
            patch.object(camem.torch, "empty", side_effect=mock_torch_empty),
        ):
            allocator.sleep(offload_tags="tag1")

            # only offload tag1, other tag2 call unmap_and_release
            assert data1.cpu_backup_tensor is not None
            assert data2.cpu_backup_tensor is None
            mock_unmap.assert_any_call(handle1)
            mock_unmap.assert_any_call(handle2)
            assert mock_unmap.call_count == 2
            assert mock_memcpy.called

    def test_wake_up_loads_and_clears_cpu_backup(self):
        camem = load_camem_module()
        allocator = camem.CaMemAllocator.get_instance()
        mock_memcpy = MagicMock()
        mock_create_and_map = MagicMock()

        handle = (1, 10, 1000, 0)
        tensor = torch.zeros(5, dtype=torch.uint8)
        data = camem.AllocationData(handle, "tag1", cpu_backup_tensor=tensor)
        allocator.pointer_to_data = {1000: data}

        with (
            patch.object(camem, "create_and_map", mock_create_and_map),
            patch.object(camem, "memcpy", mock_memcpy),
        ):
            allocator.wake_up(tags=["tag1"])

            mock_create_and_map.assert_called_once_with(handle)
            assert data.cpu_backup_tensor is None
            assert mock_memcpy.called

    def test_reload_does_not_skip_late_acl_candidate(self, tmp_path, monkeypatch):
        first_site_packages = tmp_path / "first" / "python" / "site-packages"
        second_site_packages = tmp_path / "second" / "python" / "site-packages"
        first_site_packages.mkdir(parents=True)
        second_site_packages.mkdir(parents=True)
        (first_site_packages / "acl.py").write_text(
            "class rt:\n    pass\n",
            encoding="utf-8",
        )
        (second_site_packages / "acl.py").write_text(
            "class rt:\n    memcpy = object()\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path / "first"))
        monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(tmp_path / "second"))
        monkeypatch.setattr(
            sys,
            "path",
            [path for path in sys.path if "site-packages" not in path],
            raising=False,
        )

        with patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("acl", None)
            sys.modules.pop("acl.rt", None)
            module = load_camem_module(None)

        assert module.memcpy is not None

    def test_use_memory_pool_context_manager(self):
        camem = load_camem_module()
        allocator = camem.CaMemAllocator.get_instance()
        old_tag = allocator.current_tag

        # mock use_memory_pool_with_allocator
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = "data"
        mock_ctx.__exit__.return_value = None

        with patch.object(camem, "use_memory_pool_with_allocator", return_value=mock_ctx):
            with allocator.use_memory_pool(tag="my_tag"):
                assert allocator.current_tag == "my_tag"
            # restore old tag after context manager exits
            assert allocator.current_tag == old_tag

    def test_get_current_usage(self):
        camem = load_camem_module()
        allocator = camem.CaMemAllocator.get_instance()

        allocator.pointer_to_data = {
            1: camem.AllocationData((0, 100, 1, 0), "tag"),
            2: camem.AllocationData((0, 200, 2, 0), "tag"),
        }

        usage = allocator.get_current_usage()
        assert usage == 300
