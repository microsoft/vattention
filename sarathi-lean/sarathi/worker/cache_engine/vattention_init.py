from typing import Any, Dict


def dispatch_init_kvcache(backend: Any, init_request: Dict[str, Any]):
    init_mode = init_request["init_mode"]

    if init_mode == "legacy_dense_kv":
        return backend.init_kvcache(*init_request["legacy_args"])

    if init_mode == "component_spec":
        if not hasattr(backend, "init_kvcache_component_spec"):
            raise NotImplementedError(
                "vAttention backend does not implement component-spec initialization yet"
            )
        return backend.init_kvcache_component_spec(init_request["payload"])

    raise ValueError(f"Unsupported vAttention init mode: {init_mode}")
