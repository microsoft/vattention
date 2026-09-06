"""CacheEngine class for managing the KV cache."""
import traceback
from typing import List, Tuple, Union
from sarathi.core.datatypes.sequence import Sequence
import torch
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.config import CacheConfig, ModelConfig, ParallelConfig
from sarathi.logger import init_logger
from sarathi.model_executor.attention import get_attention_wrapper
from sarathi.utils import in_wsl
from sarathi.worker.cache_engine.base_cache_engine import BaseCacheEngine
from sarathi.worker.cache_engine.vattention_init import dispatch_init_kvcache
import vattention
from sarathi.model_executor.attention import get_attention_wrapper
from sarathi.config import CacheArchitecture
logger = init_logger(__name__)

KVCache = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]

VATTENTION_MLA_VALIDATION_PROFILES = {
    "bounded_mla_suite_v1": {
        "profile_name": "bounded_mla_suite_v1",
        "max_peak_persistent_bytes": 160,
        "min_free_blocks_overall": 5,
        "max_largest_growth_bytes": 128,
        "max_largest_reclaim_bytes": 128,
    },
    "bounded_mla_suite_relaxed": {
        "profile_name": "bounded_mla_suite_relaxed",
        "max_peak_persistent_bytes": 192,
        "min_free_blocks_overall": 4,
        "max_largest_growth_bytes": 160,
        "max_largest_reclaim_bytes": 160,
    },
}


def summarize_vattention_cache_usage(
    cache_spec,
    seq_lens,
    *,
    free_blocks=None,
    seq_to_batch_idx=None,
    scheduled_batch_indices=None,
    scheduled_prompt_batch_indices=None,
    scheduled_decode_batch_indices=None,
) -> dict:
    persistent_tokens = sum(max(int(seq_len), 0) for seq_len in seq_lens)
    active_batch_indices = tuple(
        batch_idx for batch_idx, seq_len in enumerate(seq_lens) if int(seq_len) > 0
    )
    architecture = (
        cache_spec.architecture.value
        if hasattr(cache_spec.architecture, "value")
        else str(cache_spec.architecture)
    )
    cache_components = tuple(
        component.name
        for component in getattr(cache_spec, "cache_components", ())
    )
    return {
        "architecture": architecture,
        "persistent_tokens": persistent_tokens,
        "persistent_bytes_per_token": cache_spec.cached_token_bytes_local,
        "persistent_bytes": persistent_tokens * cache_spec.cached_token_bytes_local,
        "page_buffer_token_bytes": cache_spec.page_buffer_token_bytes,
        "cache_components": cache_components,
        "uses_component_resident_cache": cache_spec.architecture == CacheArchitecture.MLA,
        "active_batch_indices": active_batch_indices,
        "active_request_count": len(active_batch_indices),
        "free_blocks": free_blocks,
        "seq_to_batch_idx": (
            dict(sorted(seq_to_batch_idx.items()))
            if seq_to_batch_idx is not None
            else None
        ),
        "scheduled_batch_indices": scheduled_batch_indices,
        "scheduled_prompt_batch_indices": scheduled_prompt_batch_indices,
        "scheduled_decode_batch_indices": scheduled_decode_batch_indices,
    }


def summarize_vattention_cache_transition(previous_usage, current_usage) -> dict:
    if previous_usage is None or current_usage is None:
        raise ValueError("previous_usage and current_usage must both be provided")

    def _delta(key):
        previous_value = previous_usage.get(key)
        current_value = current_usage.get(key)
        if previous_value is None or current_value is None:
            return None
        return current_value - previous_value

    return {
        "from_event": previous_usage.get("event"),
        "to_event": current_usage.get("event"),
        "persistent_token_delta": _delta("persistent_tokens"),
        "persistent_byte_delta": _delta("persistent_bytes"),
        "free_block_delta": _delta("free_blocks"),
        "active_request_delta": _delta("active_request_count"),
        "from_seq_to_batch_idx": previous_usage.get("seq_to_batch_idx"),
        "to_seq_to_batch_idx": current_usage.get("seq_to_batch_idx"),
        "from_active_batch_indices": previous_usage.get("active_batch_indices"),
        "to_active_batch_indices": current_usage.get("active_batch_indices"),
    }


def summarize_vattention_cache_history(history, transitions=None) -> dict:
    history = tuple(history)
    if transitions is None:
        transitions = tuple(
            summarize_vattention_cache_transition(previous_usage, current_usage)
            for previous_usage, current_usage in zip(history, history[1:])
        )
    else:
        transitions = tuple(transitions)

    if not history:
        return {
            "num_snapshots": 0,
            "num_transitions": 0,
            "peak_persistent_tokens": 0,
            "peak_persistent_bytes": 0,
            "final_persistent_tokens": 0,
            "final_persistent_bytes": 0,
            "min_free_blocks": None,
            "max_active_request_count": 0,
            "largest_growth_bytes": 0,
            "largest_reclaim_bytes": 0,
            "events": (),
        }

    persistent_tokens = [snapshot["persistent_tokens"] for snapshot in history]
    persistent_bytes = [snapshot["persistent_bytes"] for snapshot in history]
    free_blocks = [
        snapshot["free_blocks"]
        for snapshot in history
        if snapshot.get("free_blocks") is not None
    ]
    active_request_counts = [
        snapshot["active_request_count"] for snapshot in history
    ]
    byte_deltas = [
        transition["persistent_byte_delta"]
        for transition in transitions
        if transition.get("persistent_byte_delta") is not None
    ]
    growth_deltas = [delta for delta in byte_deltas if delta > 0]
    reclaim_deltas = [-delta for delta in byte_deltas if delta < 0]
    return {
        "num_snapshots": len(history),
        "num_transitions": len(transitions),
        "peak_persistent_tokens": max(persistent_tokens),
        "peak_persistent_bytes": max(persistent_bytes),
        "final_persistent_tokens": persistent_tokens[-1],
        "final_persistent_bytes": persistent_bytes[-1],
        "min_free_blocks": min(free_blocks) if free_blocks else None,
        "max_active_request_count": max(active_request_counts),
        "largest_growth_bytes": max(growth_deltas) if growth_deltas else 0,
        "largest_reclaim_bytes": max(reclaim_deltas) if reclaim_deltas else 0,
        "events": tuple(snapshot.get("event") for snapshot in history),
    }


def summarize_vattention_cache_sweeps(pattern_summaries) -> dict:
    pattern_summaries = tuple(pattern_summaries)
    if not pattern_summaries:
        return {
            "num_patterns": 0,
            "pattern_names": (),
            "max_peak_persistent_bytes": 0,
            "max_peak_persistent_tokens": 0,
            "min_free_blocks_overall": None,
            "max_largest_growth_bytes": 0,
            "max_largest_reclaim_bytes": 0,
            "pattern_with_max_peak_bytes": None,
            "pattern_with_min_free_blocks": None,
        }

    def _pattern_name(summary):
        return summary.get("pattern_name")

    max_peak_summary = max(
        pattern_summaries,
        key=lambda summary: summary["peak_persistent_bytes"],
    )
    free_block_summaries = [
        summary for summary in pattern_summaries
        if summary.get("min_free_blocks") is not None
    ]
    min_free_summary = (
        min(free_block_summaries, key=lambda summary: summary["min_free_blocks"])
        if free_block_summaries
        else None
    )
    return {
        "num_patterns": len(pattern_summaries),
        "pattern_names": tuple(_pattern_name(summary) for summary in pattern_summaries),
        "max_peak_persistent_bytes": max(
            summary["peak_persistent_bytes"] for summary in pattern_summaries
        ),
        "max_peak_persistent_tokens": max(
            summary["peak_persistent_tokens"] for summary in pattern_summaries
        ),
        "min_free_blocks_overall": (
            None if min_free_summary is None else min_free_summary["min_free_blocks"]
        ),
        "max_largest_growth_bytes": max(
            summary["largest_growth_bytes"] for summary in pattern_summaries
        ),
        "max_largest_reclaim_bytes": max(
            summary["largest_reclaim_bytes"] for summary in pattern_summaries
        ),
        "pattern_with_max_peak_bytes": _pattern_name(max_peak_summary),
        "pattern_with_min_free_blocks": (
            None if min_free_summary is None else _pattern_name(min_free_summary)
        ),
    }


def summarize_vattention_cache_sweep_family(
    family_name,
    pattern_summaries,
) -> dict:
    sweep_summary = summarize_vattention_cache_sweeps(pattern_summaries)
    return {"family_name": family_name} | sweep_summary


def summarize_vattention_cache_sweep_matrix(family_summaries) -> dict:
    family_summaries = tuple(family_summaries)
    if not family_summaries:
        return {
            "num_families": 0,
            "family_names": (),
            "max_peak_persistent_bytes": 0,
            "max_peak_persistent_tokens": 0,
            "min_free_blocks_overall": None,
            "max_largest_growth_bytes": 0,
            "max_largest_reclaim_bytes": 0,
            "family_with_max_peak_bytes": None,
            "family_with_min_free_blocks": None,
        }

    def _family_name(summary):
        return summary.get("family_name")

    max_peak_summary = max(
        family_summaries,
        key=lambda summary: summary["max_peak_persistent_bytes"],
    )
    free_block_summaries = [
        summary for summary in family_summaries
        if summary.get("min_free_blocks_overall") is not None
    ]
    min_free_summary = (
        min(free_block_summaries, key=lambda summary: summary["min_free_blocks_overall"])
        if free_block_summaries
        else None
    )
    return {
        "num_families": len(family_summaries),
        "family_names": tuple(_family_name(summary) for summary in family_summaries),
        "max_peak_persistent_bytes": max(
            summary["max_peak_persistent_bytes"] for summary in family_summaries
        ),
        "max_peak_persistent_tokens": max(
            summary["max_peak_persistent_tokens"] for summary in family_summaries
        ),
        "min_free_blocks_overall": (
            None if min_free_summary is None else min_free_summary["min_free_blocks_overall"]
        ),
        "max_largest_growth_bytes": max(
            summary["max_largest_growth_bytes"] for summary in family_summaries
        ),
        "max_largest_reclaim_bytes": max(
            summary["max_largest_reclaim_bytes"] for summary in family_summaries
        ),
        "family_with_max_peak_bytes": _family_name(max_peak_summary),
        "family_with_min_free_blocks": (
            None if min_free_summary is None else _family_name(min_free_summary)
        ),
    }


def validate_vattention_cache_sweep_matrix(
    matrix_summary,
    *,
    max_peak_persistent_bytes=None,
    min_free_blocks_overall=None,
    max_largest_growth_bytes=None,
    max_largest_reclaim_bytes=None,
):
    violations = []

    def _check_upper_bound(key, expected):
        if expected is None:
            return
        observed = matrix_summary.get(key)
        if observed is not None and observed > expected:
            violations.append(
                {
                    "metric": key,
                    "constraint": "<=",
                    "expected": expected,
                    "observed": observed,
                }
            )

    def _check_lower_bound(key, expected):
        if expected is None:
            return
        observed = matrix_summary.get(key)
        if observed is None or observed < expected:
            violations.append(
                {
                    "metric": key,
                    "constraint": ">=",
                    "expected": expected,
                    "observed": observed,
                }
            )

    _check_upper_bound("max_peak_persistent_bytes", max_peak_persistent_bytes)
    _check_upper_bound("max_largest_growth_bytes", max_largest_growth_bytes)
    _check_upper_bound("max_largest_reclaim_bytes", max_largest_reclaim_bytes)
    _check_lower_bound("min_free_blocks_overall", min_free_blocks_overall)

    return {
        "is_valid": not violations,
        "violations": tuple(violations),
    }


def summarize_vattention_cache_validation_suite(matrix_summaries) -> dict:
    matrix_summaries = tuple(matrix_summaries)
    if not matrix_summaries:
        return {
            "num_matrices": 0,
            "matrix_names": (),
            "max_peak_persistent_bytes": 0,
            "min_free_blocks_overall": None,
            "max_largest_growth_bytes": 0,
            "max_largest_reclaim_bytes": 0,
            "matrix_with_max_peak_bytes": None,
            "matrix_with_min_free_blocks": None,
        }

    def _matrix_name(summary):
        return summary.get("matrix_name")

    max_peak_summary = max(
        matrix_summaries,
        key=lambda summary: summary["max_peak_persistent_bytes"],
    )
    free_block_summaries = [
        summary for summary in matrix_summaries
        if summary.get("min_free_blocks_overall") is not None
    ]
    min_free_summary = (
        min(free_block_summaries, key=lambda summary: summary["min_free_blocks_overall"])
        if free_block_summaries
        else None
    )
    return {
        "num_matrices": len(matrix_summaries),
        "matrix_names": tuple(_matrix_name(summary) for summary in matrix_summaries),
        "max_peak_persistent_bytes": max(
            summary["max_peak_persistent_bytes"] for summary in matrix_summaries
        ),
        "min_free_blocks_overall": (
            None if min_free_summary is None else min_free_summary["min_free_blocks_overall"]
        ),
        "max_largest_growth_bytes": max(
            summary["max_largest_growth_bytes"] for summary in matrix_summaries
        ),
        "max_largest_reclaim_bytes": max(
            summary["max_largest_reclaim_bytes"] for summary in matrix_summaries
        ),
        "matrix_with_max_peak_bytes": _matrix_name(max_peak_summary),
        "matrix_with_min_free_blocks": (
            None if min_free_summary is None else _matrix_name(min_free_summary)
        ),
    }


def validate_vattention_cache_validation_suite(
    suite_summary,
    *,
    max_peak_persistent_bytes=None,
    min_free_blocks_overall=None,
    max_largest_growth_bytes=None,
    max_largest_reclaim_bytes=None,
):
    violations = []

    def _check_upper_bound(key, expected):
        if expected is None:
            return
        observed = suite_summary.get(key)
        if observed is not None and observed > expected:
            violations.append(
                {
                    "metric": key,
                    "constraint": "<=",
                    "expected": expected,
                    "observed": observed,
                }
            )

    def _check_lower_bound(key, expected):
        if expected is None:
            return
        observed = suite_summary.get(key)
        if observed is None or observed < expected:
            violations.append(
                {
                    "metric": key,
                    "constraint": ">=",
                    "expected": expected,
                    "observed": observed,
                }
            )

    _check_upper_bound("max_peak_persistent_bytes", max_peak_persistent_bytes)
    _check_upper_bound("max_largest_growth_bytes", max_largest_growth_bytes)
    _check_upper_bound("max_largest_reclaim_bytes", max_largest_reclaim_bytes)
    _check_lower_bound("min_free_blocks_overall", min_free_blocks_overall)

    return {
        "is_valid": not violations,
        "violations": tuple(violations),
    }


def compare_vattention_cache_validation_suite_to_profile(
    suite_summary,
    expected_profile,
):
    expected_profile = dict(expected_profile)
    validation = validate_vattention_cache_validation_suite(
        suite_summary,
        max_peak_persistent_bytes=expected_profile.get("max_peak_persistent_bytes"),
        min_free_blocks_overall=expected_profile.get("min_free_blocks_overall"),
        max_largest_growth_bytes=expected_profile.get("max_largest_growth_bytes"),
        max_largest_reclaim_bytes=expected_profile.get("max_largest_reclaim_bytes"),
    )
    return {
        "profile_name": expected_profile.get("profile_name"),
        "suite_summary": suite_summary,
        "expected_profile": expected_profile,
        "is_valid": validation["is_valid"],
        "violations": validation["violations"],
    }


def get_vattention_mla_validation_profile(profile_name):
    if profile_name not in VATTENTION_MLA_VALIDATION_PROFILES:
        raise KeyError(f"Unknown vAttention MLA validation profile: {profile_name}")
    return dict(VATTENTION_MLA_VALIDATION_PROFILES[profile_name])


def list_vattention_mla_validation_profiles():
    return tuple(VATTENTION_MLA_VALIDATION_PROFILES.keys())


def compare_vattention_cache_validation_suite_to_named_profile(
    suite_summary,
    profile_name,
):
    return compare_vattention_cache_validation_suite_to_profile(
        suite_summary,
        get_vattention_mla_validation_profile(profile_name),
    )


def compare_vattention_cache_validation_suite_to_named_profiles(
    suite_summary,
    profile_names=None,
):
    profile_names = (
        list_vattention_mla_validation_profiles()
        if profile_names is None
        else tuple(profile_names)
    )
    return tuple(
        compare_vattention_cache_validation_suite_to_named_profile(
            suite_summary,
            profile_name,
        )
        for profile_name in profile_names
    )


def select_vattention_cache_validation_profile(
    suite_summary,
    profile_names=None,
):
    for report in compare_vattention_cache_validation_suite_to_named_profiles(
        suite_summary,
        profile_names=profile_names,
    ):
        if report["is_valid"]:
            return report
    return None


def recommend_vattention_cache_validation_profile(
    suite_summary,
    *,
    preferred_profile="bounded_mla_suite_v1",
    fallback_profiles=None,
):
    if fallback_profiles is None:
        fallback_profiles = tuple(
            profile_name
            for profile_name in list_vattention_mla_validation_profiles()
            if profile_name != preferred_profile
        )

    preferred_report = compare_vattention_cache_validation_suite_to_named_profile(
        suite_summary,
        preferred_profile,
    )
    if preferred_report["is_valid"]:
        return {
            "status": "ready",
            "selected_profile": preferred_profile,
            "selected_report": preferred_report,
            "checked_reports": (preferred_report,),
        }

    fallback_reports = compare_vattention_cache_validation_suite_to_named_profiles(
        suite_summary,
        profile_names=fallback_profiles,
    )
    for report in fallback_reports:
        if report["is_valid"]:
            return {
                "status": "relaxed_only",
                "selected_profile": report["profile_name"],
                "selected_report": report,
                "checked_reports": (preferred_report,) + tuple(fallback_reports),
            }

    return {
        "status": "blocked",
        "selected_profile": None,
        "selected_report": None,
        "checked_reports": (preferred_report,) + tuple(fallback_reports),
    }


def format_vattention_gpu_cache(cache_spec, kv_cache, device) -> List[object]:
    if cache_spec.architecture == CacheArchitecture.MLA:
        from sarathi.model_executor.models.deepseek_v2 import (
            DeepseekV2ComponentMLAKVCache,
        )

        num_q_heads_local = getattr(
            getattr(cache_spec, "tp_attention", None),
            "num_q_heads_local",
            getattr(cache_spec, "num_heads", None),
        )
        if num_q_heads_local is None:
            raise AttributeError(
                "MLA cache spec must expose tp_attention.num_q_heads_local or num_heads"
            )
        if len(kv_cache) == 2:
            kv_latent_cache, k_rope_cache = kv_cache
            assert kv_latent_cache.device == device, (
                "kv_latent cache device mismatch. expected: {}, got: {}".format(
                    device, kv_latent_cache.device
                )
            )
            assert k_rope_cache.device == device, (
                "k_rope cache device mismatch expected: {}, got: {}".format(
                    device, k_rope_cache.device
                )
            )
            return [
                DeepseekV2ComponentMLAKVCache(
                    kv_latent=kv_latent_cache[:, :, layer_idx, :],
                    k_rope=k_rope_cache[:, :, layer_idx, :],
                )
                for layer_idx in range(cache_spec.num_layers)
            ]

        if len(kv_cache) != 2 * cache_spec.num_layers:
            raise ValueError(
                "Unexpected MLA cache tensor layout from vAttention backend: "
                f"expected 2 or {2 * cache_spec.num_layers} tensors, got {len(kv_cache)}"
            )

        kv_latent_caches = kv_cache[: cache_spec.num_layers]
        k_rope_caches = kv_cache[cache_spec.num_layers :]
        for layer_idx, (kv_latent_cache, k_rope_cache) in enumerate(
            zip(kv_latent_caches, k_rope_caches)
        ):
            assert kv_latent_cache.device == device, (
                "kv_latent cache device mismatch for layer {}. expected: {}, got: {}".format(
                    layer_idx, device, kv_latent_cache.device
                )
            )
            assert k_rope_cache.device == device, (
                "k_rope cache device mismatch for layer {}. expected: {}, got: {}".format(
                    layer_idx, device, k_rope_cache.device
                )
            )
        return [
            DeepseekV2ComponentMLAKVCache(
                kv_latent=kv_latent_cache,
                k_rope=k_rope_cache,
            )
            for kv_latent_cache, k_rope_cache in zip(kv_latent_caches, k_rope_caches)
        ]

    if cache_spec.megacache:
        k_cache = kv_cache[0]
        v_cache = kv_cache[1]
        assert k_cache.device == device, \
                    "k_cache device mismatch. expected: {}, got: {}".format(device, k_cache.device)
        assert v_cache.device == device, \
                    "v_cache device mismatch expected: {}, got: {}".format(device, v_cache.device)

        return [(k_cache[:, :, i], v_cache[:, :, i]) for i in range(cache_spec.num_layers)]

    k_cache = kv_cache[:cache_spec.num_layers]
    v_cache = kv_cache[cache_spec.num_layers:]
    for i in range(cache_spec.num_layers):
        assert k_cache[i].device == device, \
                    "k_cache device mismatch. expected: {}, got: {}".format(device, k_cache[i].device)
        assert v_cache[i].device == device, \
                    "v_cache device mismatch expected: {}, got: {}".format(device, v_cache[i].device)
    return list(zip(k_cache, v_cache))

class vATTNCacheEngine(BaseCacheEngine):
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU KV cache.
    """
    _instance = None

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        mem_alloc_backend: str,
    ) -> None:
        self.max_batch_size = cache_config.max_batch_size
        self.device = torch.empty(1).cuda().device if not in_wsl() else torch.device("cuda")
        self.device_idx = int(str(self.device).split(":")[-1])
        self.max_model_seq_len = model_config.max_model_len
        self.curr_seq_lens = [0 for i in range(self.max_batch_size)]
        self.seq_to_batch_idx = {}
        self.curr_batch_idx = None
        self.prompt_batch_indices = ()
        self.decode_batch_indices = ()
        self.cache_usage_history = []
        self.page_size = cache_config.page_size
        self.vattn_async = True if mem_alloc_backend == "async" else False
        self.vattn_mega_cache = True if "megacache" in model_config.attention_backend.lower() else False
        self.cache_mem_size = cache_config.memory_for_gpu
        self.init_spec = model_config.get_vattention_init_spec(
            page_size=self.page_size,
            parallel_config=parallel_config,
            megacache=self.vattn_mega_cache,
            max_batch_size=self.max_batch_size,
            max_context_length=self.max_model_seq_len,
            device_idx=self.device_idx,
        )
        self.cache_spec = self.init_spec.cache_spec
        super().__init__(cache_config, model_config, parallel_config)

    def num_free_blocks(self) -> int:
        return vattention.num_free_kvblocks()

    def _init_kvcache_from_spec(self):
        return dispatch_init_kvcache(
            vattention,
            self.init_spec.get_extension_init_request(),
        )

    def allocate_gpu_cache(self) -> List[torch.Tensor]:
        print(f"\n[PYTHON TRACE] Initializing KV Cache:")
        print(f" > Architecture: {self.cache_spec.architecture.value}")
        print(f" > Layers: {self.num_layers}, Heads: {self.num_heads}, Head Size: {self.head_size}")
        print(f" > Max Batch: {self.max_batch_size}, Max Seq: {self.max_model_seq_len}")
        print(f" > MegaCache Enabled: {self.vattn_mega_cache}")
        print(f" > Tokens Per Page: {self.cache_spec.tokens_per_page}")
        print(f" > Page Buffer Token Bytes: {self.cache_spec.page_buffer_token_bytes}")
        
        kv_cache = self._init_kvcache_from_spec()
        cache_list = format_vattention_gpu_cache(self.cache_spec, kv_cache, self.device)
        
        print(f"[PYTHON TRACE] Reserving Physical Memory: {self.cache_mem_size / (1024**2):.2f} MB")
        vattention.reserve_physical_pages(self.cache_mem_size)
        return cache_list

    def preempt_requests(self, preempted_seq: List[int]) -> None:
        for seq in preempted_seq:
            self.free_request(seq.seq_id)

    def get_k_cache(self, layer_idx: int) -> torch.Tensor:
        return self.gpu_cache[layer_idx][0]

    def get_v_cache(self, layer_idx: int) -> torch.Tensor:
        return self.gpu_cache[layer_idx][1]

    def get_request_allocator_metrics(self, seq_id: int) -> dict | None:
        batch_idx = self.seq_to_batch_idx.get(seq_id)
        if batch_idx is None:
            return None

        seq_len = int(self.curr_seq_lens[batch_idx])
        if seq_len <= 0:
            return None

        mapped_blocks = int(vattention.debug_request_mapped_blocks(batch_idx))
        metrics = dict(vattention.debug_fragmentation_metrics(seq_len, mapped_blocks))
        return {
            "mapped_blocks": mapped_blocks,
            "fragmentation_percent": float(metrics["token_frag_pct"]),
        }
    
    def step(self, seq_metadata_list: List[SequenceMetadata]) -> None:
        b_idx_prompt = []
        b_idx_gen = []
        for seq_metadata in seq_metadata_list:
            
            if seq_metadata.is_prompt:
                seq_id = seq_metadata.seq.seq_id
                prompt_chunk_len = seq_metadata.prompt_chunk_len
                current_prompt_chunk_len = seq_metadata.seq.get_next_prompt_chunk_len(
                prompt_chunk_len
                )
                processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_processed()

                context_len = processed_prompt_len + current_prompt_chunk_len
                new_batch_idx = self.get_req_batch_idx(seq_id, context_len)
                self.curr_seq_lens[new_batch_idx] = context_len
                # b_idx.append(new_batch_idx)
                b_idx_prompt.append(new_batch_idx)
            
            else:
                context_len = seq_metadata.seq.get_len()
                seq_id = seq_metadata.seq.seq_id
                new_batch_idx = self.get_req_batch_idx(seq_id, context_len)
                self.curr_seq_lens[new_batch_idx] = context_len 
                # b_idx.append(new_batch_idx)
                b_idx_gen.append(new_batch_idx)

        if self.vattn_async:
            vattention.step_async(self.curr_seq_lens)
        else:
            vattention.step(self.curr_seq_lens, True)

        self.prompt_batch_indices = tuple(b_idx_prompt)
        self.decode_batch_indices = tuple(b_idx_gen)
        self.curr_batch_idx = torch.tensor(b_idx_prompt+b_idx_gen, dtype=torch.int32, device=self.device)
        get_attention_wrapper().set_batch_idx(self.curr_batch_idx, torch.tensor(b_idx_gen, dtype=torch.int32, device=self.device))
        self._record_cache_usage_snapshot("step")

    def on_step_completion(self, seq_metadata_list: List[SequenceMetadata]) -> None:
        for seq_metadata in seq_metadata_list:
            if seq_metadata.seq.is_finished():
                self.free_request(seq_metadata.seq.seq_id)

    def get_req_batch_idx(self, seq_id: int, seq_len: int) -> int:
        if seq_id in self.seq_to_batch_idx:
            return self.seq_to_batch_idx[seq_id]

        return self.alloc_new_batch_idx(seq_id, seq_len)

    def alloc_new_batch_idx(self, seq_id: int, seq_len: int) -> int:
        new_batch_idx = vattention.alloc_new_batch_idx(seq_len)
        if new_batch_idx == -1:
            print(self.curr_seq_lens)
        assert new_batch_idx != -1, "Failed to allocate new batch idx. This is not expected..."
        self.seq_to_batch_idx[seq_id] = new_batch_idx
        return new_batch_idx

    def free_request(self, seq_id: int) -> None:
        if seq_id in self.seq_to_batch_idx:
            batch_idx = self.seq_to_batch_idx[seq_id]
            vattention.free_batch_idx(batch_idx)
            self.seq_to_batch_idx.pop(seq_id)
            self.curr_seq_lens[batch_idx] = 0
            self._record_cache_usage_snapshot("free_request")
            return
        raise Exception(f"seq_id {seq_id} not found in req_table")

    def reclaim_req_ids(self) -> None:
        for seq_id in list(self.seq_to_batch_idx.keys()):
            self.free_request(seq_id)

    def get_batch_idx(self) -> torch.Tensor:
        return self.curr_batch_idx

    def clear_batch_index(self) -> None:
        self.curr_batch_idx = None

    def release_kvcache_physical(self):
        vattention.release_kvcache_physical()

    def disable_deferred_reclamation(self):
        vattention.set_deferred_reclamation(False)

    def get_attention_context_lens(self):
        return self.attn_context_lens

    def get_cache_usage_stats(self) -> dict:
        return summarize_vattention_cache_usage(
            self.cache_spec,
            self.curr_seq_lens,
            free_blocks=self.num_free_blocks(),
            seq_to_batch_idx=self.seq_to_batch_idx,
            scheduled_batch_indices=(
                None
                if getattr(self, "curr_batch_idx", None) is None
                else tuple(self.curr_batch_idx.tolist())
            ),
            scheduled_prompt_batch_indices=getattr(self, "prompt_batch_indices", None),
            scheduled_decode_batch_indices=getattr(self, "decode_batch_indices", None),
        )

    def _record_cache_usage_snapshot(self, event: str) -> None:
        if not hasattr(self, "cache_spec") or not hasattr(self, "curr_seq_lens"):
            return
        snapshot = dict(self.get_cache_usage_stats())
        snapshot["event"] = event
        if not hasattr(self, "cache_usage_history"):
            self.cache_usage_history = []
        self.cache_usage_history.append(snapshot)

    def get_cache_usage_history(self):
        return tuple(getattr(self, "cache_usage_history", ()))

    def get_cache_usage_transitions(self):
        history = self.get_cache_usage_history()
        return tuple(
            summarize_vattention_cache_transition(previous_usage, current_usage)
            for previous_usage, current_usage in zip(history, history[1:])
        )

    def get_cache_usage_summary(self):
        history = self.get_cache_usage_history()
        transitions = self.get_cache_usage_transitions()
        return summarize_vattention_cache_history(history, transitions)

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        megacache = "megacache" in model_config.attention_backend.lower()
        return model_config.get_vattention_cache_block_size_bytes(
            block_size,
            parallel_config,
            megacache=megacache,
        )

    def cleanup_kvcache(self):
        # this is required to ensure UVM module is not holding on to the memory
        vattention.cleanup()
