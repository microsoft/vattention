from sarathi.benchmark.request_generator.gamma_request_interval_generator import (
    GammaRequestIntervalGenerator,
)
from sarathi.benchmark.request_generator.poisson_request_interval_generator import (
    PoissonRequestIntervalGenerator,
)
from sarathi.benchmark.request_generator.static_request_interval_generator import (
    StaticRequestIntervalGenerator,
)
from sarathi.benchmark.request_generator.trace_request_interval_generator import (
    TraceRequestIntervalGenerator,
)
from sarathi.benchmark.types import RequestIntervalGeneratorType
from sarathi.utils.base_registry import BaseRegistry


class RequestIntervalGeneratorRegistry(BaseRegistry):

    @classmethod
    def get_key_from_str(cls, key_str: str) -> RequestIntervalGeneratorType:
        return RequestIntervalGeneratorType.from_str(key_str)


RequestIntervalGeneratorRegistry.register(
    RequestIntervalGeneratorType.GAMMA, GammaRequestIntervalGenerator
)
RequestIntervalGeneratorRegistry.register(
    RequestIntervalGeneratorType.POISSON, PoissonRequestIntervalGenerator
)
RequestIntervalGeneratorRegistry.register(
    RequestIntervalGeneratorType.STATIC, StaticRequestIntervalGenerator
)
RequestIntervalGeneratorRegistry.register(
    RequestIntervalGeneratorType.TRACE, TraceRequestIntervalGenerator
)
