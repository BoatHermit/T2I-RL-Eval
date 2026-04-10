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
