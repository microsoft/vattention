import json
from pathlib import Path
from typing import Optional

from transformers import AutoConfig, PretrainedConfig

from sarathi.transformers_utils.configs import *  # pylint: disable=wildcard-import

_CONFIG_REGISTRY = {
    "deepseek_v2": DeepseekV2Config,
    "qwen": QWenConfig,
    "RefinedWeb": RWConfig,  # For tiiuae/falcon-40b(-instruct)
    "RefinedWebModel": RWConfig,  # For tiiuae/falcon-7b(-instruct)
    "yi": YiConfig,
}


def _register_custom_configs() -> None:
    register = getattr(AutoConfig, "register", None)
    if register is None:
        return
    for model_type, config_class in _CONFIG_REGISTRY.items():
        try:
            register(model_type, config_class)
        except ValueError:
            # Another import path may have registered the same config already.
            continue


def _load_known_local_config(
    model: str,
    revision: Optional[str] = None,
) -> Optional[PretrainedConfig]:
    config_path = Path(model) / "config.json"
    if not config_path.exists():
        return None
    config_payload = json.loads(config_path.read_text())
    model_type = config_payload.get("model_type")
    config_class = _CONFIG_REGISTRY.get(model_type)
    if config_class is None:
        return None
    return config_class.from_pretrained(model, revision=revision)


def get_config(
    model: str, trust_remote_code: bool, revision: Optional[str] = None
) -> PretrainedConfig:
    _register_custom_configs()
    try:
        config = AutoConfig.from_pretrained(
            model, trust_remote_code=trust_remote_code, revision=revision
        )
    except ValueError as e:
        local_config = _load_known_local_config(model, revision=revision)
        if local_config is not None:
            return local_config
        if (
            not trust_remote_code
            and "requires you to execute the configuration file" in str(e)
        ):
            err_msg = (
                "Failed to load the model config. If the model is a custom "
                "model not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e
    if config.model_type in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[config.model_type]
        config = config_class.from_pretrained(model, revision=revision)
    return config
