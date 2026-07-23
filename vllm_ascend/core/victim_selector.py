from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.request import Request

_DEFAULT_MAX_TOKENS = 1024


@dataclass(frozen=True)
class UtilityVictimSelectorConfig:
    enable_utility_victim_selection: bool = False
    utility_kill_switch: bool = False
    utility_completion_weight: float = 0.5
    utility_preempt_weight: float = 0.3
    utility_kv_gate: float = 0.0
    utility_cooldown_s: float = 0.0
    utility_min_running: int = 1
    utility_snapshot_enabled: bool = False
    utility_snapshot_top_k: int = 3
    utility_snapshot_history_size: int = 32
    utility_epsilon: float = 1e-6
    utility_default_max_tokens: int = _DEFAULT_MAX_TOKENS

    @classmethod
    def from_additional_config(
        cls,
        additional_config: dict[str, Any] | None,
    ) -> UtilityVictimSelectorConfig:
        config_data = additional_config or {}
        defaults = cls()
        config = cls(
            enable_utility_victim_selection=bool(
                config_data.get(
                    "enable_utility_victim_selection",
                    defaults.enable_utility_victim_selection,
                )
            ),
            utility_kill_switch=bool(config_data.get("utility_kill_switch", defaults.utility_kill_switch)),
            utility_completion_weight=float(
                config_data.get("utility_completion_weight", defaults.utility_completion_weight)
            ),
            utility_preempt_weight=float(config_data.get("utility_preempt_weight", defaults.utility_preempt_weight)),
            utility_kv_gate=float(config_data.get("utility_kv_gate", defaults.utility_kv_gate)),
            utility_cooldown_s=float(config_data.get("utility_cooldown_s", defaults.utility_cooldown_s)),
            utility_min_running=int(config_data.get("utility_min_running", defaults.utility_min_running)),
            utility_snapshot_enabled=bool(
                config_data.get("utility_snapshot_enabled", defaults.utility_snapshot_enabled)
            ),
            utility_snapshot_top_k=int(config_data.get("utility_snapshot_top_k", defaults.utility_snapshot_top_k)),
            utility_snapshot_history_size=int(
                config_data.get(
                    "utility_snapshot_history_size",
                    defaults.utility_snapshot_history_size,
                )
            ),
            utility_epsilon=float(config_data.get("utility_epsilon", defaults.utility_epsilon)),
            utility_default_max_tokens=int(
                config_data.get("utility_default_max_tokens", defaults.utility_default_max_tokens)
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.utility_completion_weight < 0:
            raise ValueError("utility_completion_weight must be non-negative")
        if self.utility_preempt_weight < 0:
            raise ValueError("utility_preempt_weight must be non-negative")
        if self.utility_kv_gate < 0 or self.utility_kv_gate > 1:
            raise ValueError("utility_kv_gate must be in [0, 1]")
        if self.utility_cooldown_s < 0:
            raise ValueError("utility_cooldown_s must be non-negative")
        if self.utility_min_running <= 0:
            raise ValueError("utility_min_running must be positive")
        if self.utility_snapshot_top_k <= 0:
            raise ValueError("utility_snapshot_top_k must be positive")
        if self.utility_snapshot_history_size <= 0:
            raise ValueError("utility_snapshot_history_size must be positive")
        if self.utility_epsilon <= 0:
            raise ValueError("utility_epsilon must be positive")
        if self.utility_default_max_tokens <= 0:
            raise ValueError("utility_default_max_tokens must be positive")

    def to_additional_config(self) -> dict[str, Any]:
        return {
            "enable_utility_victim_selection": self.enable_utility_victim_selection,
            "utility_kill_switch": self.utility_kill_switch,
            "utility_completion_weight": self.utility_completion_weight,
            "utility_preempt_weight": self.utility_preempt_weight,
            "utility_kv_gate": self.utility_kv_gate,
            "utility_cooldown_s": self.utility_cooldown_s,
            "utility_min_running": self.utility_min_running,
            "utility_snapshot_enabled": self.utility_snapshot_enabled,
            "utility_snapshot_top_k": self.utility_snapshot_top_k,
            "utility_snapshot_history_size": self.utility_snapshot_history_size,
            "utility_epsilon": self.utility_epsilon,
            "utility_default_max_tokens": self.utility_default_max_tokens,
        }

    @classmethod
    def from_vllm_config(cls, vllm_config) -> UtilityVictimSelectorConfig:
        additional_config = getattr(vllm_config, "additional_config", None) or {}
        return cls.from_additional_config(additional_config)


@dataclass(frozen=True)
class UtilityCandidateScore:
    request_id: str
    utility: float
    evict_score: float
    tokens_freed: int
    completion: float
    num_preemptions: int
    arrival_time: float


class UnifiedVictimSelector:
    def __init__(self, config: UtilityVictimSelectorConfig) -> None:
        self.config = config
        self._last_utility_pick_ts = -math.inf
        snapshot_size = max(1, int(self.config.utility_snapshot_history_size))
        self._total_preemptions = 0
        self._total_tokens_freed = 0
        self._kv_pressure_events = 0
        self._utility_strategy_hits = 0
        self._default_strategy_hits = 0
        self._consecutive_preemption_events = 0
        self._consecutive_preemption_checks = 0
        self._last_preempted_request_id: str | None = None
        self._preemptions_per_request: dict[str, int] = defaultdict(int)
        self._recent_preempted_req_ids: deque[str] = deque(maxlen=snapshot_size)
        self._decision_snapshots: deque[dict[str, Any]] = deque(maxlen=snapshot_size)

    @classmethod
    def from_vllm_config(cls, vllm_config) -> UnifiedVictimSelector:
        return cls(UtilityVictimSelectorConfig.from_vllm_config(vllm_config))

    def pick_victim(
        self,
        running: Sequence[Request],
        policy: SchedulingPolicy,
        *,
        kv_utilization: float | None = None,
        now_s: float | None = None,
    ) -> Request:
        if not running:
            raise ValueError("running is empty, cannot pick victim")

        now = self._resolve_now(now_s)
        if self.config.utility_kv_gate > 0 and kv_utilization is not None:
            if kv_utilization >= self.config.utility_kv_gate:
                self._kv_pressure_events += 1

        default_victim = self._pick_default_victim(running, policy)
        ranked_candidates: list[UtilityCandidateScore] = []
        req_map: dict[str, Request] = {}
        utility_enabled = self._utility_enabled and self._can_use_utility(
            kv_utilization=kv_utilization,
            now_s=now,
            running_size=len(running),
        )

        if utility_enabled:
            ranked_candidates, req_map = self._rank_candidates(running)
            victim = req_map[ranked_candidates[0].request_id]
            self._last_utility_pick_ts = now
        else:
            victim = default_victim
            if self.config.utility_snapshot_enabled:
                ranked_candidates, _ = self._rank_candidates(running)

        self._record_preemption(
            victim=victim,
            used_utility=utility_enabled,
            policy=policy,
            kv_utilization=kv_utilization,
            now_s=now,
            default_victim=default_victim,
            ranked_candidates=ranked_candidates,
            running_size=len(running),
        )
        return victim

    @property
    def _utility_enabled(self) -> bool:
        return self.config.enable_utility_victim_selection and not self.config.utility_kill_switch

    @staticmethod
    def _pick_default_victim(running: Sequence[Request], policy: SchedulingPolicy) -> Request:
        if policy == SchedulingPolicy.PRIORITY:
            return max(
                running,
                key=lambda request: (request.priority, request.arrival_time),
            )
        return running[-1]

    def _can_use_utility(
        self,
        *,
        kv_utilization: float | None,
        now_s: float | None,
        running_size: int,
    ) -> bool:
        if running_size < self.config.utility_min_running:
            return False

        if self.config.utility_kv_gate > 0:
            if kv_utilization is None or kv_utilization < self.config.utility_kv_gate:
                return False

        if self.config.utility_cooldown_s > 0 and self._last_utility_pick_ts > -math.inf:
            now = self._resolve_now(now_s)
            if now - self._last_utility_pick_ts < self.config.utility_cooldown_s:
                return False

        return True

    @staticmethod
    def _resolve_now(now_s: float | None) -> float:
        if now_s is not None:
            return float(now_s)
        return time.monotonic()

    def _rank_candidates(self, running: Sequence[Request]) -> tuple[list[UtilityCandidateScore], dict[str, Request]]:
        req_map: dict[str, Request] = {}
        candidates: list[UtilityCandidateScore] = []
        for request in running:
            request_id = str(getattr(request, "request_id", ""))
            req_map[request_id] = request
            candidates.append(self._score_request(request, request_id))

        # BidKV semantics: higher utility should be preempted first.
        # Keep deterministic tie-breakers by arrival/request id.
        candidates.sort(key=lambda c: (-c.utility, c.arrival_time, c.request_id))
        return candidates, req_map

    def _score_request(self, request: Request, request_id: str) -> UtilityCandidateScore:
        tokens_freed = max(int(getattr(request, "num_computed_tokens", 0) or 0), 0)
        completion = self._compute_completion(request)
        num_preemptions = max(int(getattr(request, "num_preemptions", 0) or 0), 0)
        arrival_time = float(getattr(request, "arrival_time", 0.0) or 0.0)
        utility, evict_score = self._compute_utility(
            tokens_freed=tokens_freed,
            completion=completion,
            num_preemptions=num_preemptions,
        )
        return UtilityCandidateScore(
            request_id=request_id,
            utility=utility,
            evict_score=evict_score,
            tokens_freed=tokens_freed,
            completion=completion,
            num_preemptions=num_preemptions,
            arrival_time=arrival_time,
        )

    def _compute_utility(
        self,
        *,
        tokens_freed: int,
        completion: float,
        num_preemptions: int,
    ) -> tuple[float, float]:
        reward = max(float(tokens_freed), 0.0)
        preemptions = max(float(num_preemptions), 0.0)

        delta = (
            1.0 + self.config.utility_completion_weight * completion + self.config.utility_preempt_weight * preemptions
        )
        utility = reward / max(delta + self.config.utility_epsilon, self.config.utility_epsilon)
        evict_score = utility
        return utility, evict_score

    def _compute_completion(self, request: Request) -> float:
        output_tokens = self._output_tokens(request)
        max_tokens = getattr(request, "max_tokens", None)
        if not isinstance(max_tokens, (int, float)) or max_tokens <= 0:
            max_tokens = self.config.utility_default_max_tokens

        completion = float(output_tokens) / float(max_tokens)
        return min(max(completion, 0.0), 1.0)

    @staticmethod
    def _output_tokens(request: Request) -> int:
        output_token_ids = getattr(request, "output_token_ids", None)
        if output_token_ids is not None:
            try:
                return len(output_token_ids)
            except TypeError:
                pass
        return int(getattr(request, "num_output_tokens", 0) or 0)

    def _record_preemption(
        self,
        *,
        victim: Request,
        used_utility: bool,
        policy: SchedulingPolicy,
        kv_utilization: float | None,
        now_s: float,
        default_victim: Request,
        ranked_candidates: Sequence[UtilityCandidateScore],
        running_size: int,
    ) -> None:
        request_id = str(getattr(victim, "request_id", ""))
        tokens_freed = max(int(getattr(victim, "num_computed_tokens", 0) or 0), 0)

        self._total_preemptions += 1
        self._total_tokens_freed += tokens_freed
        if request_id and self._last_preempted_request_id is not None:
            self._consecutive_preemption_checks += 1
            if request_id == self._last_preempted_request_id:
                self._consecutive_preemption_events += 1

        if request_id:
            self._preemptions_per_request[request_id] += 1
            self._recent_preempted_req_ids.append(request_id)
            self._last_preempted_request_id = request_id

        if used_utility:
            self._utility_strategy_hits += 1
        else:
            self._default_strategy_hits += 1

        if self.config.utility_snapshot_enabled:
            top_k = max(1, int(self.config.utility_snapshot_top_k))
            selected_id = request_id
            default_id = str(getattr(default_victim, "request_id", ""))
            snapshot_candidates = [
                {
                    "rank": index + 1,
                    "request_id": candidate.request_id,
                    "utility": round(candidate.utility, 6),
                    "evict_score": round(candidate.evict_score, 6),
                    "tokens_freed": candidate.tokens_freed,
                    "completion": round(candidate.completion, 6),
                    "num_preemptions": candidate.num_preemptions,
                    "arrival_time": candidate.arrival_time,
                    "selected": candidate.request_id == selected_id,
                }
                for index, candidate in enumerate(ranked_candidates[:top_k])
            ]
            self._decision_snapshots.append(
                {
                    "timestamp_s": round(now_s, 6),
                    "policy": getattr(policy, "name", str(policy)),
                    "used_utility": used_utility,
                    "kv_utilization": kv_utilization,
                    "running_size": running_size,
                    "selected_victim_id": selected_id,
                    "default_victim_id": default_id,
                    "candidates": snapshot_candidates,
                }
            )

    def export_metrics(self) -> dict[str, Any]:
        hit_rate = 0.0
        if self._total_preemptions > 0:
            hit_rate = self._utility_strategy_hits / self._total_preemptions

        consecutive_preempt_ratio = 0.0
        if self._consecutive_preemption_checks > 0:
            consecutive_preempt_ratio = self._consecutive_preemption_events / self._consecutive_preemption_checks

        return {
            "total_preemptions": self._total_preemptions,
            "total_tokens_freed": self._total_tokens_freed,
            "kv_pressure_events": self._kv_pressure_events,
            "consecutive_preempt_ratio": consecutive_preempt_ratio,
            "preemptions_per_request_p95": self._percentile(list(self._preemptions_per_request.values()), 95),
            "preempted_req_ids": list(self._recent_preempted_req_ids),
            "strategy_hit_rate": hit_rate,
            "utility_strategy_hits": self._utility_strategy_hits,
            "default_strategy_hits": self._default_strategy_hits,
        }

    def get_recent_snapshots(self, limit: int = 10) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        return list(self._decision_snapshots)[-limit:]

    def emit_observability_log(self, logger, scheduler_name: str) -> None:
        metrics = self.export_metrics()
        if metrics["total_preemptions"] <= 0:
            return

        logger.info(
            "[UtilityVictim][%s] total_preemptions=%d utility_hits=%d default_hits=%d "
            "hit_rate=%.3f tokens_freed=%d kv_pressure_events=%d "
            "consecutive_preempt_ratio=%.3f p95_preemptions_per_request=%.2f",
            scheduler_name,
            metrics["total_preemptions"],
            metrics["utility_strategy_hits"],
            metrics["default_strategy_hits"],
            metrics["strategy_hit_rate"],
            metrics["total_tokens_freed"],
            metrics["kv_pressure_events"],
            metrics["consecutive_preempt_ratio"],
            metrics["preemptions_per_request_p95"],
        )

        if self.config.utility_snapshot_enabled:
            snapshots = self.get_recent_snapshots(limit=1)
            if snapshots:
                logger.debug("[UtilityVictim][%s] latest_snapshot=%s", scheduler_name, snapshots[0])

    @staticmethod
    def _percentile(values: Sequence[int], percentile: int) -> float:
        data = sorted(int(v) for v in values if v is not None)
        if not data:
            return 0.0

        rank = math.ceil((percentile / 100.0) * len(data)) - 1
        rank = max(0, min(rank, len(data) - 1))
        return float(data[rank])


def infer_kv_utilization_from_scheduler(scheduler) -> float | None:
    try:
        block_pool = scheduler.kv_cache_manager.block_pool
        total_blocks = float(getattr(block_pool, "num_gpu_blocks", 0) or 0)
        if total_blocks <= 0:
            return None

        free_block_queue = getattr(block_pool, "free_block_queue", None)
        free_blocks = float(getattr(free_block_queue, "num_free_blocks", 0) or 0)
        used_ratio = (total_blocks - free_blocks) / total_blocks
        return min(max(used_ratio, 0.0), 1.0)
    except AttributeError:
        return None
