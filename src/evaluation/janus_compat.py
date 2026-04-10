from __future__ import annotations

import dataclasses
import functools
from typing import Any


def _patch_mutable_defaults_on_config_class(cls: type[Any]) -> bool:
    patched = False
    for field_name in getattr(cls, "__annotations__", {}):
        if not hasattr(cls, field_name):
            continue
        current_value = getattr(cls, field_name)
        if isinstance(current_value, dict):
            setattr(cls, field_name, dataclasses.field(default_factory=dict))
            patched = True
        elif isinstance(current_value, list):
            setattr(cls, field_name, dataclasses.field(default_factory=list))
            patched = True
        elif isinstance(current_value, set):
            setattr(cls, field_name, dataclasses.field(default_factory=set))
            patched = True
    return patched


def _patch_missing_post_init_on_model_class(cls: type[Any]) -> bool:
    if getattr(cls, "__codex_post_init_patched__", False):
        return False

    post_init = getattr(cls, "post_init", None)
    original_init = getattr(cls, "__init__", None)
    if post_init is None or original_init is None:
        return False

    @functools.wraps(original_init)
    def _patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, "all_tied_weights_keys"):
            self.post_init()

    cls.__init__ = _patched_init
    cls.__codex_post_init_patched__ = True
    return True


def import_janus_vlchatprocessor() -> type[Any]:
    from transformers import configuration_utils as transformers_configuration_utils

    original_dataclass = transformers_configuration_utils.dataclass

    def _safe_dataclass(cls=None, **kwargs):
        if cls is None:
            return lambda wrapped_cls: _safe_dataclass(wrapped_cls, **kwargs)
        try:
            return original_dataclass(cls, **kwargs)
        except ValueError as exc:
            if "mutable default" not in str(exc):
                raise
            if not _patch_mutable_defaults_on_config_class(cls):
                raise
            return original_dataclass(cls, **kwargs)

    transformers_configuration_utils.dataclass = _safe_dataclass
    try:
        from janus.models import MultiModalityCausalLM, VLChatProcessor
    finally:
        transformers_configuration_utils.dataclass = original_dataclass
    _patch_missing_post_init_on_model_class(MultiModalityCausalLM)
    return VLChatProcessor
