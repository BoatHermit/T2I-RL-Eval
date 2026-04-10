from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.janus_compat import (
    _patch_missing_post_init_on_model_class,
    _patch_mutable_defaults_on_config_class,
    build_janus_model_load_kwargs,
    build_janus_retry_load_kwargs,
    is_meta_tensor_item_error,
)


def test_patch_mutable_defaults_on_config_class_rewrites_dict_list_and_set_defaults():
    class FakeConfig:
        params: dict = {}
        names: list = []
        tags: set = set()
        stable_value: int = 1

    patched = _patch_mutable_defaults_on_config_class(FakeConfig)

    assert patched is True
    assert isinstance(FakeConfig.params, dataclasses.Field)
    assert isinstance(FakeConfig.names, dataclasses.Field)
    assert isinstance(FakeConfig.tags, dataclasses.Field)
    assert FakeConfig.stable_value == 1


def test_patch_mutable_defaults_on_config_class_skips_classes_without_mutable_defaults():
    class FakeConfig:
        stable_value: int = 1
        label: str = "ok"

    patched = _patch_mutable_defaults_on_config_class(FakeConfig)

    assert patched is False
    assert FakeConfig.stable_value == 1
    assert FakeConfig.label == "ok"


def test_patch_missing_post_init_on_model_class_calls_post_init_when_needed():
    class FakeModel:
        def __init__(self, value: int):
            self.value = value
            self.post_init_calls = 0

        def post_init(self):
            self.post_init_calls += 1
            self.all_tied_weights_keys = {}

    patched = _patch_missing_post_init_on_model_class(FakeModel)
    model = FakeModel(3)

    assert patched is True
    assert model.value == 3
    assert model.post_init_calls == 1
    assert model.all_tied_weights_keys == {}


def test_patch_missing_post_init_on_model_class_skips_post_init_when_already_initialized():
    class FakeModel:
        def __init__(self):
            self.post_init_calls = 0
            self.all_tied_weights_keys = {"weight": "base.weight"}

        def post_init(self):
            self.post_init_calls += 1

    patched = _patch_missing_post_init_on_model_class(FakeModel)
    model = FakeModel()

    assert patched is True
    assert model.post_init_calls == 0
    assert model.all_tied_weights_keys == {"weight": "base.weight"}


def test_build_janus_model_load_kwargs_disables_meta_init_for_full_precision_loads():
    load_kwargs = build_janus_model_load_kwargs(torch_dtype="float16")

    assert load_kwargs == {
        "trust_remote_code": True,
        "torch_dtype": "float16",
        "low_cpu_mem_usage": False,
    }


def test_build_janus_model_load_kwargs_keeps_device_map_for_quantized_loads():
    quant_config = object()

    load_kwargs = build_janus_model_load_kwargs(
        torch_dtype="float16",
        quantization_config=quant_config,
    )

    assert load_kwargs == {
        "trust_remote_code": True,
        "torch_dtype": "float16",
        "quantization_config": quant_config,
        "device_map": "auto",
    }


def test_build_janus_retry_load_kwargs_removes_device_map_and_forces_concrete_init():
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": "float16",
        "quantization_config": object(),
        "device_map": "auto",
    }

    retry_kwargs = build_janus_retry_load_kwargs(load_kwargs)

    assert retry_kwargs["trust_remote_code"] is True
    assert retry_kwargs["torch_dtype"] == "float16"
    assert "device_map" not in retry_kwargs
    assert retry_kwargs["low_cpu_mem_usage"] is False


def test_is_meta_tensor_item_error_matches_runtime_message():
    exc = RuntimeError("Tensor.item() cannot be called on meta tensors")

    assert is_meta_tensor_item_error(exc) is True
    assert is_meta_tensor_item_error(RuntimeError("different failure")) is False
