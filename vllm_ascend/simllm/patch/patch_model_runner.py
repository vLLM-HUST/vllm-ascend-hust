#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

"""Sim-LLM patch for ``NPUModelRunner.execute_model()``.

When ``VLLM_ASCEND_SIMLLM_ENABLED=1``, wraps ``execute_model``,
``_model_forward``, and the attention backend's ``do_kv_cache_update`` to
implement inter-task KV reuse with real acceleration:

**For matched requests (≈similarity above threshold):**
  1. Preprocess from ``SchedulerOutput`` (before batch is built).
  2. Identify cached KV matches.
  3. Rewrite ``num_computed_tokens`` → ``_prepare_inputs`` only schedules
     the last token → **true prefill skip** (1-token decode instead of
     full prefill).
  4. Hijacked ``do_kv_cache_update`` injects cached top-layer KV into ALL
     layers via the normal write path.

**For unique requests (no match above threshold):**
  5. ``_simllm_apply_sandwich_slots()`` sets ``slot_mapping=-1`` on MIDDLE
     layers so only ``keep_layers`` (bottom-N + top-N) cache KV → ~81%
     BlockTable memory savings.

**Post-forward (all requests):**
  6. Extract KV: matched → top-layer only; unique → keep_layers average.
     Store in KVManager for future matching.

Patches three targets: ``execute_model``, ``_model_forward``, and the
Ascend / FlashAttention ``do_kv_cache_update``.

Follows the same monkey-patch pattern used for DeepSeek, Qwen, and other
model-specific patches in ``vllm_ascend/patch/worker/``.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any

import regex as re
import torch
from vllm.forward_context import get_forward_context

from vllm_ascend.simllm.config import SimLLMConfig
from vllm_ascend.simllm.kv_reuse import KVReuseEngine
from vllm_ascend.simllm.utils import (
    cumsum_to_ranges,
    resolve_input_embedding_dim,
    tensor_to_int_list,
    tensor_to_int_matrix,
)

logger = logging.getLogger(__name__)

# Regex to parse layer index from attention layer names like
# "model.layers.5.self_attn".
_LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")

# ---------------------------------------------------------------------------
# Module-level singletons — initialised once at patch-apply time, reused
# across every execute_model / _model_forward call within the worker process.
# ---------------------------------------------------------------------------
_simllm_config: SimLLMConfig | None = None
_kv_manager: Any = None  # KVManager
_simhash_hasher: Any = None  # SimHashHasher
_similarity_identifier: Any = None  # SimilarityIdentifier
_sandwich_config: Any = None  # SandwichConfig
_kv_reuse_engine: Any = None  # KVReuseEngine
_original_execute_model: Any = None
_original_model_forward: Any = None

# Per-forward injection map — built before _model_forward, consumed by the
# hijacked do_kv_cache_update inside every attention layer.
#   dict[batch_idx → (k_flat, v_flat, tok_start, covered)]
# where k_flat / v_flat have shape [L_kv, num_kv_heads, head_size].
_simllm_injection_map: dict[int, tuple] | None = None


def _patch_do_kv_cache_update() -> None:
    """Monkey-patch attention-backend ``do_kv_cache_update`` methods.

    Replaces matched-token slices of *key* / *value* with cached KV so that
    ``reshape_and_cache`` writes injected KV into the cache through the
    normal path.  All layers get the same top-layer cached KV.

    Patches both the Ascend NPU backend and the CUDA FlashAttention backend
    so Sim-LLM works on either hardware.
    """
    global _original_ascend_kv_update, _original_flash_kv_update

    # -- Ascend NPU backend ------------------------------------------------
    try:
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

        _original_ascend_kv_update = AscendAttentionBackendImpl.do_kv_cache_update

        def _ascend_kv_update(
            self_impl: Any,
            layer: Any,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: Any,
            slot_mapping: torch.Tensor,
        ) -> None:
            _inject_into_kv(key, value)
            _original_ascend_kv_update(
                self_impl,
                layer,
                key,
                value,
                kv_cache,
                slot_mapping,
            )

        AscendAttentionBackendImpl.do_kv_cache_update = _ascend_kv_update  # type: ignore[assignment]
        logger.info("SimLLM: patched AscendAttentionBackendImpl.do_kv_cache_update.")
    except Exception:
        logger.debug("SimLLM: Ascend attention backend not available, skipping.")

    # -- CUDA FlashAttention backend ---------------------------------------
    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

        _original_flash_kv_update = FlashAttentionImpl.do_kv_cache_update

        def _flash_kv_update(
            self_impl: Any,
            layer: Any,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> None:
            _inject_into_kv(key, value)
            _original_flash_kv_update(
                self_impl,
                layer,
                key,
                value,
                kv_cache,
                slot_mapping,
            )

        FlashAttentionImpl.do_kv_cache_update = _flash_kv_update  # type: ignore[method-assign]
        logger.info("SimLLM: patched FlashAttentionImpl.do_kv_cache_update.")
    except Exception:
        logger.debug("SimLLM: FlashAttention backend not available, skipping.")


def _inject_into_kv(key: torch.Tensor, value: torch.Tensor) -> None:
    """Replace matched-token slices of *key* / *value* with cached KV.

    Called from hijacked ``do_kv_cache_update`` in every attention layer.
    *key* / *value* have shape ``[num_tokens, num_kv_heads, head_size]``.
    """
    global _simllm_injection_map
    inj_map = _simllm_injection_map
    if inj_map is None:
        return
    for _batch_idx, (k_flat, v_flat, tok_start, covered) in inj_map.items():
        # k_flat: [L_kv, H, D] — same shape as key[tok_start:tok_start+covered]
        if covered > 0:
            key[tok_start : tok_start + covered] = k_flat.to(
                device=key.device,
                dtype=key.dtype,
                non_blocking=True,
            )
            value[tok_start : tok_start + covered] = v_flat.to(
                device=value.device,
                dtype=value.dtype,
                non_blocking=True,
            )


# Stash originals so tests can restore them.
_original_ascend_kv_update: Any = None
_original_flash_kv_update: Any = None


def apply_simllm_patch(model_runner_cls: Any | None = None) -> None:
    """Apply the Sim-LLM patch to NPUModelRunner.

    Called once per worker after ``NPUModelRunner`` is defined.
    When ``VLLM_ASCEND_SIMLLM_ENABLED=0`` this is a silent no-op.

    Patches both ``execute_model`` (for proactive matching/rewrite) and
    ``_model_forward`` (for KV injection / extraction at the right point
    in the execution pipeline).
    """
    global _simllm_config, _kv_manager, _simhash_hasher
    global _similarity_identifier, _sandwich_config, _kv_reuse_engine
    global _original_execute_model, _original_model_forward

    config = SimLLMConfig.from_env()
    if not config.enabled:
        return

    if model_runner_cls is None:
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        model_runner_cls = NPUModelRunner

    if getattr(model_runner_cls, "execute_model", None) is _simllm_execute_model:
        return

    logger.info("Applying Sim-LLM patch to NPUModelRunner …")

    _simllm_config = config

    from vllm_ascend.simllm.kv_manager import KVManager
    from vllm_ascend.simllm.kv_reuse import KVReuseEngine
    from vllm_ascend.simllm.lsh import SimHashHasher
    from vllm_ascend.simllm.sandwich import SandwichConfig
    from vllm_ascend.simllm.similarity import SimilarityIdentifier

    _kv_manager = KVManager(max_cache_size=config.kv_cache_size)
    _simhash_hasher = SimHashHasher(
        dim=4096,  # default for Qwen2.5-7B; overridden after model load
        num_bits=config.lsh_num_bits,
    )
    _similarity_identifier = SimilarityIdentifier(
        cosine_threshold=config.cosine_threshold,
        lsh_batch_threshold=config.lsh_batch_threshold,
        lsh_num_bits=config.lsh_num_bits,
    )
    _sandwich_config = SandwichConfig(
        bottom_layers=config.sandwich_bottom,
        top_layers=config.sandwich_top,
    )
    _kv_reuse_engine = KVReuseEngine(
        block_size=128,  # Ascend 910B optimal; overridden from kv_cache_config later
        num_kv_heads=8,  # overridden after model load
        head_size=128,  # overridden after model load
    )

    # Patch attention backend's do_kv_cache_update to inject cached KV.
    _patch_do_kv_cache_update()

    # Patch execute_model (lightweight — triggers the full pipeline).
    _original_execute_model = model_runner_cls.execute_model
    model_runner_cls.execute_model = _simllm_execute_model  # type: ignore[method-assign]

    # Patch _model_forward (heavy lifting — inject/extract KV at the right time).
    _original_model_forward = model_runner_cls._model_forward
    model_runner_cls._model_forward = _simllm_model_forward  # type: ignore[method-assign]

    logger.info(
        "Sim-LLM patch applied (cache_size=%d, threshold=%.2f, sandwich_bottom=%d, sandwich_top=%d, lsh_bits=%d).",
        config.kv_cache_size,
        config.cosine_threshold,
        config.sandwich_bottom,
        config.sandwich_top,
        config.lsh_num_bits,
    )


# ===========================================================================
# Proactive preprocessing — runs BEFORE the original execute_model so we
# can rewrite scheduler_output and avoid full prefill for matched requests.
# ===========================================================================


def _simllm_preprocess_from_scheduler(self: Any, scheduler_output: Any) -> None:
    """Extract embeddings and LSH hashes from *scheduler_output* directly.

    Runs before ``_original_execute_model`` so we can identify matches and
    rewrite ``num_computed_tokens`` before the batch is built.  Does NOT
    depend on ``input_batch`` (which is stale / not yet populated).
    """
    new_reqs = scheduler_output.scheduled_new_reqs
    if not new_reqs:
        self._simllm_batch_embeddings = None
        self._simllm_batch_hashes = None
        self._simllm_batch_req_ids = None
        return

    try:
        _reconcile_hasher_dim(self)

        # Build flat input_ids and query_start_loc from scheduler_output.
        all_ids: list[int] = []
        qsl = [0]
        req_ids: list[str] = []
        for req in new_reqs:
            req_ids.append(req.req_id)
            ids = req.prompt_token_ids or []
            all_ids.extend(ids)
            qsl.append(qsl[-1] + len(ids))

        if not all_ids:
            self._simllm_batch_embeddings = None
            self._simllm_batch_hashes = None
            self._simllm_batch_req_ids = None
            return

        input_ids = torch.tensor(all_ids, device=self.device)
        query_start_loc = torch.tensor(qsl, device=self.device)

        from vllm_ascend.simllm.hooks.preprocess import SimLLMPreprocessor

        preprocessor = SimLLMPreprocessor(
            pooling=_simllm_config.embedding_pooling,  # type: ignore[union-attr]
        )
        embeddings = preprocessor.extract_embeddings(
            self.model,
            input_ids,
            query_start_loc,
        )

        hashes = _simhash_hasher.hash(embeddings)  # type: ignore[misc]

        self._simllm_batch_embeddings = embeddings
        self._simllm_batch_hashes = hashes
        self._simllm_batch_req_ids = req_ids

    except Exception:
        logger.exception("SimLLM preprocess_from_scheduler failed — processing as unmatched.")
        self._simllm_batch_embeddings = None
        self._simllm_batch_hashes = None
        self._simllm_batch_req_ids = None


def _simllm_rewrite_scheduler_output(self: Any, scheduler_output: Any) -> None:
    """Modify ``num_computed_tokens`` for matched requests.

    For each matched request, set ``num_computed_tokens`` so that
    ``_prepare_inputs`` treats the covered tokens as already cached and
    only schedules the minimal remaining tokens (ideally 1 — decode mode).

    ``NewRequestData`` is a mutable dataclass, so we modify its
    ``num_computed_tokens`` field in place.
    """
    match_results = getattr(self, "_simllm_match_results", None)
    if not match_results:
        return

    new_reqs = scheduler_output.scheduled_new_reqs

    rewritten = 0
    for batch_idx, m in match_results.items():
        if not m.matched or m.cached_k is None:
            continue
        if batch_idx >= len(new_reqs):
            continue

        req = new_reqs[batch_idx]
        prompt_len = len(req.prompt_token_ids or [])
        cached_len = m.cached_k.shape[2]  # L_kv in [1, H, L, D]
        covered = min(cached_len, prompt_len)

        # Keep at least 1 token for the model to process (generates logits).
        if covered <= 1:
            continue

        req.num_computed_tokens = covered - 1
        rewritten += 1

    if rewritten:
        logger.debug(
            "SimLLM rewrite_scheduler: skipped prefill for %d matched requests (avg coverage=%d tokens).",
            rewritten,
            sum(
                min(
                    match_results[i].cached_k.shape[2],  # type: ignore[union-attr]
                    len(new_reqs[i].prompt_token_ids or []),
                )
                for i in match_results
                if i < len(new_reqs) and match_results[i].matched and match_results[i].cached_k is not None
            )
            // max(rewritten, 1),
        )


def _simllm_build_injection_map_from_scheduler(self: Any, scheduler_output: Any) -> None:
    """Build the injection map from *scheduler_output* token positions.

    Called after ``_simllm_rewrite_scheduler_output`` so the covered-token
    ranges are aligned with the modified ``num_computed_tokens`` values.
    """
    global _simllm_injection_map
    _simllm_injection_map = None

    match_results = getattr(self, "_simllm_match_results", None)
    if not match_results:
        return

    _reconcile_kv_reuse_engine(self)

    new_reqs = scheduler_output.scheduled_new_reqs

    # Build query_start_loc from (possibly rewritten) scheduler_output.
    qsl = [0]
    for req in new_reqs:
        qsl.append(qsl[-1] + len(req.prompt_token_ids or []))

    inj_map: dict[int, tuple] = {}
    for batch_idx, m in match_results.items():
        if not m.matched or m.cached_k is None:
            continue
        if batch_idx >= len(new_reqs):
            continue

        req = new_reqs[batch_idx]
        prompt_len = len(req.prompt_token_ids or [])
        cached_len = m.cached_k.shape[2]
        covered = min(cached_len, prompt_len)
        if covered == 0:
            continue

        # Align + flatten cached KV.
        k_aligned, v_aligned = _kv_reuse_engine.prepare_injection(  # type: ignore[misc]
            m.cached_k,
            m.cached_v,
            covered,
        )
        k_flat = k_aligned.squeeze(0).permute(1, 0, 2).contiguous()
        v_flat = v_aligned.squeeze(0).permute(1, 0, 2).contiguous()
        tok_start = qsl[batch_idx]

        inj_map[batch_idx] = (k_flat, v_flat, tok_start, covered)

    if inj_map:
        _simllm_injection_map = inj_map


def _parse_layer_idx(layer_name: str) -> int | None:
    """Extract layer index from an attention layer name, e.g. ``"model.layers.5.self_attn"`` → 5."""
    m = _LAYER_IDX_RE.search(layer_name)
    return int(m.group(1)) if m else None


def _simllm_apply_sandwich_slots(self: Any) -> None:
    """Set ``slot_mapping=-1`` on MIDDLE layers for UNMATCHED requests.

    Middle layers (not in ``keep_layers``) skip KV cache writes, saving
    ~81% of BlockTable memory per unique request.  Only ``keep_layers``
    (bottom-N + top-N) retain KV for future matching.
    """
    match_results = getattr(self, "_simllm_match_results", {})
    num_reqs = self.input_batch.num_reqs
    if num_reqs == 0:
        return

    # Build set of UNMATCHED batch indices.
    unmatched = {i for i in range(num_reqs) if i not in match_results or not match_results[i].matched}
    if not unmatched:
        return

    try:
        ctx = get_forward_context()
        slot_mapping_dict = ctx.slot_mapping
    except Exception:
        logger.debug("SimLLM sandwich: forward context not available, skipping.")
        return

    if not isinstance(slot_mapping_dict, dict):
        return  # spec-decode list path — skip for now

    keep_layers = _sandwich_config.keep_layers  # type: ignore[union-attr]

    qsl = self.query_start_loc
    if hasattr(qsl, "gpu"):
        query_start_loc = qsl.gpu[: num_reqs + 1]
    else:
        query_start_loc = qsl[: num_reqs + 1]
    seq_lens = self.seq_lens[:num_reqs]
    query_ranges = cumsum_to_ranges(query_start_loc)
    seq_len_values = tensor_to_int_list(seq_lens)

    disabled = 0
    for layer_name, sm_tensor in slot_mapping_dict.items():
        layer_idx = _parse_layer_idx(layer_name)
        if layer_idx is None or layer_idx in keep_layers:
            continue

        for batch_idx in unmatched:
            s_len = seq_len_values[batch_idx]
            if s_len == 0:
                continue
            tok_start, _ = query_ranges[batch_idx]
            tok_end = tok_start + s_len
            sm_tensor[tok_start:tok_end] = -1
        disabled += 1

    if disabled:
        logger.debug(
            "SimLLM sandwich: disabled KV cache for %d middle layers (%d unique requests, keep_layers=%s).",
            disabled,
            len(unmatched),
            sorted(keep_layers),
        )


# ===========================================================================
# execute_model wrapper — proactive preprocessing + scheduler rewrite
# ===========================================================================


def _simllm_execute_model(
    self: Any,
    scheduler_output: Any,
    intermediate_tensors: Any = None,
    **kwargs: Any,
) -> Any:
    """Wrapped ``NPUModelRunner.execute_model`` with proactive preprocessing.

    Identifies matched requests BEFORE the batch is built so we can rewrite
    ``num_computed_tokens`` and skip prefill for matched requests.
    """
    if not _simllm_config or not _simllm_config.enabled:
        return _original_execute_model(self, scheduler_output, intermediate_tensors, **kwargs)

    # -- Phase 0: Preprocess + identify from scheduler_output -------------
    _simllm_preprocess_from_scheduler(self, scheduler_output)
    self._simllm_match_results = _simllm_identify(self)

    # -- Phase 0b: Rewrite scheduler_output for matched requests ----------
    _simllm_rewrite_scheduler_output(self, scheduler_output)
    _simllm_build_injection_map_from_scheduler(self, scheduler_output)

    self._simllm_scheduler_output = scheduler_output
    self._simllm_deferrals = set()

    # -- Original execute_model (sees modified num_computed_tokens) -------
    outputs = _original_execute_model(self, scheduler_output, intermediate_tensors, **kwargs)

    _simllm_handle_deferrals(self)
    return outputs


# ===========================================================================
# _model_forward wrapper — sandwich + extract (preprocess moved to exec)
# ===========================================================================


def _simllm_model_forward(
    self: Any,
    num_tokens_padded: int,
    input_ids: Any = None,
    positions: Any = None,
    intermediate_tensors: Any = None,
    inputs_embeds: Any = None,
    **model_kwargs: Any,
) -> Any:
    """Patched ``_model_forward`` — sandwich slots + KV extraction.

    Preprocessing and identification now happen in ``_simllm_execute_model``
    (before this is called).  Here we only:
    1. Apply sandwich slot protection for unique requests.
    2. Run the original forward (hijacked ``do_kv_cache_update`` injects KV).
    3. Extract KV from cache for storage in KVManager.
    """
    global _simllm_injection_map

    if not _simllm_config or not _simllm_config.enabled:
        return _original_model_forward(
            self,
            num_tokens_padded,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **model_kwargs,
        )

    # -- Sandwich: disable KV cache for middle layers (unique requests) ---
    _simllm_apply_sandwich_slots(self)

    # -- Original forward (hijacked do_kv_cache_update injects KV) --------
    try:
        hidden_states = _original_model_forward(
            self,
            num_tokens_padded,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **model_kwargs,
        )
    finally:
        _simllm_injection_map = None

    # -- Extract KV + store in KVManager ----------------------------------
    _simllm_extract_kv(self, hidden_states)

    return hidden_states


# ===========================================================================
# Hook implementations
# ===========================================================================


def _simllm_preprocess(self: Any) -> None:
    """Extract per-request embeddings via token-embedding layer + LSH hash.

    Called inside patched ``_model_forward`` — by this point
    ``self.input_batch`` is fully populated by ``_prepare_inputs()``.
    """
    num_reqs = self.input_batch.num_reqs
    if num_reqs == 0:
        self._simllm_batch_embeddings = None
        self._simllm_batch_hashes = None
        self._simllm_batch_req_ids = None
        return

    try:
        # Reconcile SimHashHasher dimension with actual model embedding dim.
        _reconcile_hasher_dim(self)

        # Access populated input data.
        num_tokens = self.input_batch.num_tokens[:num_reqs].sum()
        if num_tokens == 0:
            self._simllm_batch_embeddings = None
            self._simllm_batch_hashes = None
            self._simllm_batch_req_ids = None
            return

        input_ids = self.input_batch.input_ids[:num_tokens]

        qsl = self.query_start_loc
        if hasattr(qsl, "gpu"):
            query_start_loc = qsl.gpu[: num_reqs + 1]
        else:
            query_start_loc = qsl[: num_reqs + 1]

        from vllm_ascend.simllm.hooks.preprocess import SimLLMPreprocessor

        preprocessor = SimLLMPreprocessor(
            pooling=_simllm_config.embedding_pooling,  # type: ignore[union-attr]
        )
        embeddings = preprocessor.extract_embeddings(self.model, input_ids, query_start_loc)  # [num_reqs, D]

        hashes = _simhash_hasher.hash(embeddings)  # type: ignore[misc]

        self._simllm_batch_embeddings = embeddings
        self._simllm_batch_hashes = hashes
        self._simllm_batch_req_ids = list(self.input_batch.req_ids[:num_reqs])

    except Exception:
        logger.exception("SimLLM preprocess failed — falling back to normal forward.")
        self._simllm_batch_embeddings = None
        self._simllm_batch_hashes = None
        self._simllm_batch_req_ids = None


def _simllm_identify(self: Any) -> dict[int, Any]:
    """Match batch embeddings against cached tasks in KVManager."""
    embeddings = getattr(self, "_simllm_batch_embeddings", None)
    hashes = getattr(self, "_simllm_batch_hashes", None)

    if embeddings is None or hashes is None or embeddings.shape[0] == 0:
        return {}

    try:
        from vllm_ascend.simllm.hooks.identify import identify_batch

        return identify_batch(
            embeddings,
            hashes,
            _kv_manager,  # type: ignore[misc]
            _similarity_identifier,  # type: ignore[misc]
        )
    except Exception:
        logger.exception("SimLLM identify failed — processing all as unmatched.")
        return {}


def _simllm_inject_kv(self: Any) -> None:
    """Write cached top-layer KV into BlockTable for matched requests.

    Legacy/test-support helper from the earlier pre-population path.  The
    current primary path injects K/V inside the hijacked ``do_kv_cache_update``.

    Injects the same cached top-layer KV into the **top-N layers only**
    (controlled by ``SIMLLM_SANDWICH_TOP``, default 3).  Bottom/middle
    layers compute their own K,V normally from hidden states.  This
    reduces injection overhead by ~90% while preserving the semantic
    benefit — top-layer KV carries the most transferable information.

    Runs inside patched ``_model_forward`` — ``block_table`` is already
    committed and ``self.kv_caches`` is bound at this point.
    """
    match_results = getattr(self, "_simllm_match_results", None)
    if not match_results:
        return

    if not hasattr(self, "kv_caches") or not self.kv_caches:
        logger.debug("SimLLM inject_kv: kv_caches not available yet, skipping.")
        return

    try:
        num_reqs = self.input_batch.num_reqs
        blk_table = self.input_batch.block_table[0]
        blk_table_tensor = blk_table.get_device_tensor()
        block_size = _kv_reuse_engine._block_size

        _reconcile_kv_reuse_engine(self)

        # Determine target layers: top-N only.
        num_layers = len(self.kv_caches)
        top_n = _sandwich_config.top_layers  # type: ignore[union-attr]
        if top_n <= 0 or top_n >= num_layers:
            target_layers = self.kv_caches  # fallback: all layers
        else:
            target_layers = self.kv_caches[num_layers - top_n :]

        matched_count = 0
        for batch_idx, m in match_results.items():
            if not m.matched or batch_idx >= num_reqs:
                continue
            if m.cached_k is None or m.cached_v is None:
                continue

            seq_len = int(self.seq_lens[batch_idx].item())
            k_aligned, v_aligned = _kv_reuse_engine.prepare_injection(m.cached_k, m.cached_v, seq_len)

            num_blocks = KVReuseEngine.num_blocks_needed(seq_len, block_size)
            block_ids = blk_table_tensor[batch_idx, :num_blocks].tolist()
            if not block_ids:
                continue

            # Write cached KV into top-N layers' kv_cache at those blocks.
            for layer_kv in target_layers:
                if isinstance(layer_kv, tuple):
                    k_cache, v_cache = layer_kv
                else:
                    k_cache, v_cache = layer_kv[0], layer_kv[1]
                _kv_reuse_engine.write_to_cache(
                    k_cache,
                    v_cache,
                    block_ids,
                    k_aligned,
                    v_aligned,
                )

            matched_count += 1

        if matched_count:
            logger.debug(
                "SimLLM inject_kv: injected cached KV for %d matched requests (top-%d of %d layers).",
                matched_count,
                top_n,
                num_layers,
            )

    except Exception:
        logger.exception("SimLLM inject_kv failed — continuing with normal forward.")


def _simllm_extract_kv(self: Any, hidden_states: Any) -> None:
    """Extract KV from kv_caches and store in KVManager.

    - **Matched** requests: store top-layer KV only (symmetric with injection).
    - **Unmatched** requests: store averaged KV from ``keep_layers``
      (bottom-N + top-N per ``SandwichConfig``) for richer future matching.

    Uses hidden_states from the model forward for embedding extraction
    (more semantically rich than token embeddings).
    """
    if hidden_states is None:
        return

    num_reqs = self.input_batch.num_reqs
    if num_reqs == 0:
        return

    hashes = getattr(self, "_simllm_batch_hashes", None)
    batch_req_ids = getattr(self, "_simllm_batch_req_ids", None)
    if hashes is None or getattr(hashes, "shape", (0,))[0] == 0 or not batch_req_ids:
        logger.debug("SimLLM extract_kv: no prefill hashes/req_ids for this step, skipping.")
        return

    try:
        kv_caches = getattr(self, "kv_caches", None)
        if not kv_caches:
            logger.debug("SimLLM extract_kv: kv_caches not available, skipping.")
            return

        num_layers = len(kv_caches)

        def _kv_at_layer(layer_kv):
            """Return (k_cache, v_cache) for a layer, handling tuple/tensor."""
            if isinstance(layer_kv, tuple):
                return layer_kv[0], layer_kv[1]
            return layer_kv[0], layer_kv[1]

        # Determine which layers to gather from for unmatched tasks.
        keep_layers = _sandwich_config.keep_layers  # type: ignore[union-attr]
        # Guard: ensure keep_layers indices are within bounds.
        keep_layers = sorted({idx for idx in keep_layers if 0 <= idx < num_layers})
        if not keep_layers:
            keep_layers = [num_layers - 1]  # fallback: top layer only

        # Pre-validate all keep layers are accessible.
        keep_kv = [_kv_at_layer(kv_caches[idx]) for idx in keep_layers]
        # Use first accessible layer to determine block_size / shape.
        k_sample = keep_kv[0][0]
        block_size = k_sample.shape[1]

        blk_table = self.input_batch.block_table[0]
        blk_table_tensor = blk_table.get_device_tensor()

        # -- Gather embeddings from hidden states -----------------------
        qsl = self.query_start_loc
        if hasattr(qsl, "gpu"):
            query_start_loc = qsl.gpu[: num_reqs + 1]
        else:
            query_start_loc = qsl[: num_reqs + 1]

        embeddings = _per_request_embeddings(
            hidden_states,
            query_start_loc,
            pooling=_simllm_config.embedding_pooling,  # type: ignore[union-attr]
        )

        # -- Build CachedTask per request -------------------------------
        req_ids = list(self.input_batch.req_ids[:num_reqs])
        seq_lens = self.seq_lens[:num_reqs]
        match_results = getattr(self, "_simllm_match_results", {})
        seq_len_values = tensor_to_int_list(seq_lens)
        hash_values = tensor_to_int_list(hashes)
        block_table_rows = tensor_to_int_matrix(blk_table_tensor[:num_reqs])
        store_plan = _simllm_build_store_plan(
            req_ids,
            batch_req_ids,
            len(hash_values),
        )

        if not store_plan:
            logger.debug("SimLLM extract_kv: no current input_batch rows matched prefill req_ids, skipping.")
            return

        now = time.monotonic()
        stored = 0

        for row_idx, hash_idx in store_plan:
            s_len = seq_len_values[row_idx]
            if s_len == 0:
                continue

            num_blk = KVReuseEngine.num_blocks_needed(s_len, block_size)
            block_ids = block_table_rows[row_idx][:num_blk]
            if not block_ids:
                continue

            # Determine whether this request was matched.
            is_matched = (
                row_idx in match_results
                and match_results[row_idx].matched
                and match_results[row_idx].cached_k is not None
            )

            if is_matched:
                # Matched: store only top-layer KV (symmetric with injection).
                k_cache, v_cache = _kv_at_layer(kv_caches[-1])
                k_per_req = KVReuseEngine.gather_from_cache(
                    k_cache,
                    block_ids,
                    s_len,
                    block_size,
                )
                v_per_req = KVReuseEngine.gather_from_cache(
                    v_cache,
                    block_ids,
                    s_len,
                    block_size,
                )
            else:
                # Unmatched: average KV across keep_layers (sandwich).
                k_sum = None
                v_sum = None
                for k_cache, v_cache in keep_kv:
                    k_part = KVReuseEngine.gather_from_cache(
                        k_cache,
                        block_ids,
                        s_len,
                        block_size,
                    )
                    v_part = KVReuseEngine.gather_from_cache(
                        v_cache,
                        block_ids,
                        s_len,
                        block_size,
                    )
                    if k_sum is None:
                        k_sum = k_part
                    else:
                        k_sum.add_(k_part)
                    if v_sum is None:
                        v_sum = v_part
                    else:
                        v_sum.add_(v_part)
                assert k_sum is not None and v_sum is not None
                scale = 1.0 / len(keep_kv)
                k_per_req = k_sum.mul_(scale)
                v_per_req = v_sum.mul_(scale)

            emb = (
                embeddings[row_idx : row_idx + 1]
                if embeddings is not None
                else k_per_req.new_zeros(1, k_per_req.shape[1])
            )
            hsh = hash_values[hash_idx]

            from vllm_ascend.simllm.kv_manager import CachedTask

            task = CachedTask(
                task_id=req_ids[row_idx],
                embedding=emb,
                lsh_hash=hsh,
                top_k=k_per_req,
                top_v=v_per_req,
                last_access_time=now,
                seq_len=s_len,
            )
            _kv_manager.store(task)  # type: ignore[misc]
            stored += 1

        # -- Compute diagnostic-only deferral decisions -------------------
        from vllm_ascend.simllm.hooks.postprocess import SimLLMPostprocessor

        postprocessor = SimLLMPostprocessor(
            kv_manager=_kv_manager,  # type: ignore[misc]
            pooling=_simllm_config.embedding_pooling,  # type: ignore[union-attr]
            deferral_ratio=_simllm_config.deferral_ratio,  # type: ignore[union-attr]
            max_deferrals=_simllm_config.max_deferrals,  # type: ignore[union-attr]
        )
        self._simllm_deferrals = postprocessor.compute_deferrals(
            match_results,
            num_reqs,
        )

        if stored:
            logger.debug(
                "SimLLM extract_kv: stored %d tasks (cache size=%d, sandwich_layers=%s).",
                stored,
                _kv_manager.size(),
                keep_layers,  # type: ignore[misc]
            )

    except Exception:
        logger.exception("SimLLM extract_kv failed — KV not stored for this batch.")


def _simllm_protect_kv_slots(self: Any) -> None:
    """Set slot_mapping to -1 for tokens already covered by cached KV injection.

    Legacy/test-support helper for the earlier pre-population path.  The
    current primary path no longer calls this helper.

    Prevents ``unified_kv_cache_update`` from overwriting pre-populated
    cached KV positions inside ``self.kv_caches``.  ``flash_attn_varlen_func``
    reads from *block_table* (which is untouched), so it still finds the
    injected KV at those blocks.

    Must run inside ``_model_forward`` where ``set_forward_context`` is active
    and ``slot_mapping`` is accessible via ``get_forward_context()``.
    """
    match_results = getattr(self, "_simllm_match_results", None)
    if not match_results:
        return

    try:
        ctx = get_forward_context()
        slot_mapping_raw = ctx.slot_mapping
    except Exception:
        logger.debug("SimLLM protect_kv_slots: forward context not available, skipping.")
        return

    if slot_mapping_raw is None:
        return

    # Normalise to list-of-dicts (spec-decode path uses a list).
    if isinstance(slot_mapping_raw, list):
        mappings_list: list[dict] = slot_mapping_raw
    else:
        mappings_list = [slot_mapping_raw]

    num_reqs = self.input_batch.num_reqs

    qsl = self.query_start_loc
    if hasattr(qsl, "gpu"):
        query_start_loc = qsl.gpu[: num_reqs + 1]
    else:
        query_start_loc = qsl[: num_reqs + 1]

    seq_lens = self.seq_lens[:num_reqs]

    protected_total = 0
    matched_count = 0

    for batch_idx, m in match_results.items():
        if not m.matched or m.cached_k is None:
            continue
        if batch_idx >= num_reqs:
            continue

        cached_len = m.cached_k.shape[2]  # L_kv in [1, H, L, D]
        req_seq_len = int(seq_lens[batch_idx].item())

        # Only protect tokens that have REAL cached KV (not zero-padding).
        covered = min(cached_len, req_seq_len)
        if covered == 0:
            continue

        tok_start = int(query_start_loc[batch_idx].item())
        tok_end = tok_start + covered

        # Write -1 across every layer's slot_mapping tensor so
        # reshape_and_cache_flash skips those positions.
        for sm_dict in mappings_list:
            for sm in sm_dict.values():
                sm[tok_start:tok_end] = -1

        # Tell vLLM internals that these tokens are already cached.
        with contextlib.suppress(AttributeError, IndexError):
            self.input_batch.num_computed_tokens_cpu[batch_idx] = covered

        protected_total += covered
        matched_count += 1

    if protected_total:
        logger.debug(
            "SimLLM protect_kv_slots: protected %d token slots across %d matched requests.",
            protected_total,
            matched_count,
        )


def _simllm_handle_deferrals(self: Any) -> None:
    """Log diagnostic deferral decisions from the just-completed forward.

    Phase 3 keeps deferral as future/backlog input only.  This helper must not
    re-queue, delay, drop, or reorder requests.
    """
    deferrals: set[int] = getattr(self, "_simllm_deferrals", set())
    if deferrals:
        logger.debug(
            "SimLLM: %d tasks flagged for future deferral diagnostics; processing continues in the current batch.",
            len(deferrals),
        )


# ===========================================================================
# Internal helpers
# ===========================================================================


def _reconcile_hasher_dim(self: Any) -> None:
    """Re-create SimHashHasher if the model embedding dim differs from default."""
    global _simhash_hasher
    try:
        embed_dim = resolve_input_embedding_dim(self.model)
    except Exception:
        return
    if _simhash_hasher.dim != embed_dim:
        from vllm_ascend.simllm.lsh import SimHashHasher

        _simhash_hasher = SimHashHasher(
            dim=embed_dim,
            num_bits=_simllm_config.lsh_num_bits,  # type: ignore[union-attr]
        )
        logger.info("SimLLM: re-created SimHashHasher with dim=%d.", embed_dim)


def _simllm_build_store_plan(
    input_batch_req_ids: list[str],
    prefill_req_ids: list[str],
    num_hashes: int,
) -> list[tuple[int, int]]:
    """Map prefill hash rows to current input_batch row indices."""
    req_id_to_row = {req_id: idx for idx, req_id in enumerate(input_batch_req_ids)}
    store_plan: list[tuple[int, int]] = []
    for hash_idx, req_id in enumerate(prefill_req_ids[:num_hashes]):
        row_idx = req_id_to_row.get(req_id)
        if row_idx is not None:
            store_plan.append((row_idx, hash_idx))
    return store_plan


def _reconcile_kv_reuse_engine(self: Any) -> None:
    """Update KVReuseEngine block_size / num_kv_heads / head_size from actual caches."""
    kv_caches = getattr(self, "kv_caches", None)
    if not kv_caches:
        return
    top_kv = kv_caches[-1]
    if isinstance(top_kv, tuple):
        sample = top_kv[0]
    else:
        sample = top_kv[0]
    # sample: [num_blocks, block_size, num_kv_heads, head_size]
    bs = sample.shape[1]
    nh = sample.shape[2]
    hs = sample.shape[3]
    if _kv_reuse_engine._block_size != bs or _kv_reuse_engine._num_kv_heads != nh or _kv_reuse_engine._head_size != hs:
        _kv_reuse_engine._block_size = bs
        _kv_reuse_engine._num_kv_heads = nh
        _kv_reuse_engine._head_size = hs
        logger.debug(
            "SimLLM: KVReuseEngine reconciled — block_size=%d, kv_heads=%d, head_size=%d.",
            bs,
            nh,
            hs,
        )


def _per_request_embeddings(
    hidden_states: Any,
    query_start_loc: Any,
    pooling: str = "mean",
) -> Any | None:
    """Compute per-request L2-normalized embeddings from flat hidden states."""
    ranges = cumsum_to_ranges(query_start_loc)
    num_reqs = len(ranges)
    if num_reqs == 0:
        return None
    max_len = 0
    slices: list[Any] = []
    for start, end in ranges:
        if end > start:
            sl = hidden_states[start:end]
            slices.append(sl)
            max_len = max(max_len, sl.shape[0])
    if not slices:
        return None
    D = slices[0].shape[-1]
    padded = hidden_states.new_zeros(len(slices), max_len, D)
    for i, s in enumerate(slices):
        padded[i, : s.shape[0], :] = s
    from vllm_ascend.simllm.embedding import extract_embedding

    return extract_embedding(padded, pooling=pooling)
