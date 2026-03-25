from typing import Any, Dict


def validate_component_spec_payload(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise TypeError("component-spec payload must be a dict")

    if payload.get("init_mode") != "component_spec":
        raise ValueError("component-spec payload must declare init_mode=component_spec")

    cache_spec = payload.get("cache_spec")
    if not isinstance(cache_spec, dict):
        raise ValueError("component-spec payload must include a cache_spec dict")

    required_cache_keys = {
        "architecture",
        "megacache",
        "page_size",
        "tokens_per_page",
        "cached_token_bytes_per_layer",
        "cached_token_bytes_local",
        "page_buffer_token_bytes",
        "dtype_size",
        "num_layers",
        "num_kv_heads",
        "head_size",
        "tp_attention",
        "cache_components",
    }
    missing_cache_keys = sorted(required_cache_keys - cache_spec.keys())
    if missing_cache_keys:
        raise ValueError(
            "component-spec payload cache_spec is missing keys: "
            + ", ".join(missing_cache_keys)
        )

    cache_components = cache_spec["cache_components"]
    if not isinstance(cache_components, list) or not cache_components:
        raise ValueError("component-spec payload must include non-empty cache_components")

    component_token_dim_sum = 0
    for index, component in enumerate(cache_components):
        if not isinstance(component, dict):
            raise ValueError(f"cache component at index {index} must be a dict")

        component_name = component.get("name")
        token_dim = component.get("token_dim")
        if not component_name:
            raise ValueError(f"cache component at index {index} must have a name")
        if not isinstance(token_dim, int) or token_dim <= 0:
            raise ValueError(
                f"cache component {component_name} must have a positive integer token_dim"
            )
        component_token_dim_sum += token_dim

    dtype_size = cache_spec["dtype_size"]
    if not isinstance(dtype_size, int) or dtype_size <= 0:
        raise ValueError("cache_spec.dtype_size must be a positive integer")

    cached_token_bytes_per_layer = cache_spec["cached_token_bytes_per_layer"]
    if component_token_dim_sum * dtype_size != cached_token_bytes_per_layer:
        raise ValueError(
            "cache_spec.cache_components do not match cached_token_bytes_per_layer"
        )

    numeric_fields = (
        "page_size",
        "tokens_per_page",
        "page_buffer_token_bytes",
        "num_layers",
        "num_kv_heads",
        "head_size",
    )
    for field_name in numeric_fields:
        value = cache_spec[field_name]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"cache_spec.{field_name} must be a positive integer")

    if not isinstance(payload.get("max_batch_size"), int) or payload["max_batch_size"] <= 0:
        raise ValueError("component-spec payload max_batch_size must be positive")
    if (
        not isinstance(payload.get("max_context_length"), int)
        or payload["max_context_length"] <= 0
    ):
        raise ValueError("component-spec payload max_context_length must be positive")
    if not isinstance(payload.get("device_idx"), int) or payload["device_idx"] < 0:
        raise ValueError("component-spec payload device_idx must be non-negative")
    if not payload.get("dtype"):
        raise ValueError("component-spec payload dtype must be non-empty")


def dispatch_init_kvcache(backend: Any, init_request: Dict[str, Any]):
    init_mode = init_request["init_mode"]

    if init_mode == "legacy_dense_kv":
        return backend.init_kvcache(*init_request["legacy_args"])

    if init_mode == "component_spec":
        validate_component_spec_payload(init_request["payload"])
        if not hasattr(backend, "init_kvcache_component_spec"):
            raise NotImplementedError(
                "vAttention backend does not implement component-spec initialization yet"
            )
        return backend.init_kvcache_component_spec(init_request["payload"])

    raise ValueError(f"Unsupported vAttention init mode: {init_mode}")
