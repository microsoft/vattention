import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SARATHI_ROOT = REPO_ROOT / "sarathi-lean" / "sarathi"


def _ensure_package(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def _load_module(module_name: str, file_path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_deepseek_model_module():
    _ensure_package("sarathi", SARATHI_ROOT)
    _ensure_package("sarathi.model_executor", SARATHI_ROOT / "model_executor")
    _ensure_package(
        "sarathi.model_executor.parallel_utils",
        SARATHI_ROOT / "model_executor" / "parallel_utils",
    )
    _load_module(
        "sarathi.model_executor.parallel_utils.parallel_state",
        SARATHI_ROOT / "model_executor" / "parallel_utils" / "parallel_state.py",
    )
    return _load_module(
        "sarathi.model_executor.models.deepseek_v2",
        SARATHI_ROOT / "model_executor" / "models" / "deepseek_v2.py",
    )


deepseek_module = _load_deepseek_model_module()
apply_mlp = deepseek_module.apply_mlp
apply_moe = deepseek_module.apply_moe
make_mlp_weights = deepseek_module.make_mlp_weights
make_moe_weights = deepseek_module.make_moe_weights


class DeepseekV2MoETests(unittest.TestCase):
    def _make_mlp_weights(self, scale):
        return make_mlp_weights(
            gate_proj=torch.tensor(
                [
                    [1.0 * scale, 0.0],
                    [0.0, 1.0 * scale],
                ]
            ),
            up_proj=torch.tensor(
                [
                    [1.0 * scale, 0.0],
                    [0.0, 1.0 * scale],
                ]
            ),
            down_proj=torch.tensor(
                [
                    [1.0 * scale, 0.0],
                    [0.0, 1.0 * scale],
                ]
            ),
            hidden_size=2,
        )

    def test_make_moe_weights_validates_gate_shape(self):
        expert = self._make_mlp_weights(1.0)
        with self.assertRaises(ValueError):
            make_moe_weights(
                gate=torch.zeros(2, 3),
                experts=(expert, expert),
                hidden_size=2,
            )

    def test_apply_moe_routes_to_top_expert(self):
        hidden_states = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        expert0 = self._make_mlp_weights(1.0)
        expert1 = self._make_mlp_weights(2.0)
        moe_weights = make_moe_weights(
            gate=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            experts=(expert0, expert1),
            hidden_size=2,
        )

        output = apply_moe(hidden_states, moe_weights)

        expected = torch.cat(
            [
                apply_mlp(hidden_states[:1], expert0),
                apply_mlp(hidden_states[1:], expert1),
            ],
            dim=0,
        )
        self.assertTrue(torch.allclose(output, expected, atol=1e-6, rtol=1e-6))

    def test_apply_moe_adds_shared_expert_output(self):
        hidden_states = torch.tensor([[1.0, 1.0]])
        expert = self._make_mlp_weights(1.0)
        shared = self._make_mlp_weights(0.5)
        moe_weights = make_moe_weights(
            gate=torch.tensor([[1.0, 0.0]]),
            experts=(expert,),
            shared_experts=shared,
            hidden_size=2,
        )

        output = apply_moe(hidden_states, moe_weights)

        expected = apply_mlp(hidden_states, expert) + apply_mlp(hidden_states, shared)
        self.assertTrue(torch.allclose(output, expected, atol=1e-6, rtol=1e-6))

    def test_apply_moe_normalizes_topk_probabilities(self):
        hidden_states = torch.tensor([[1.0, 0.5]])
        expert0 = self._make_mlp_weights(1.0)
        expert1 = self._make_mlp_weights(3.0)
        moe_weights = make_moe_weights(
            gate=torch.tensor([[1.0, 0.0], [0.5, 0.0]]),
            experts=(expert0, expert1),
            top_k=2,
            hidden_size=2,
        )

        output = apply_moe(hidden_states, moe_weights)
        probs = torch.softmax(hidden_states @ moe_weights.gate.t(), dim=-1)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        expected = (
            apply_mlp(hidden_states, expert0) * probs[:, :1]
            + apply_mlp(hidden_states, expert1) * probs[:, 1:2]
        )
        self.assertTrue(torch.allclose(output, expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
