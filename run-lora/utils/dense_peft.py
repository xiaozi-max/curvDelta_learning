"""Dense PEFT infrastructure — full-rank additive adapter for peft==0.17.1.

Provides:
- DenseVariant          — abstract extension point (like LoraVariant)
- DenseLayer            — full-rank adapter layer (ΔW ∈ R^{d_out×d_in})
- DenseLinear           — concrete module replacing nn.Linear
- DenseConfig           — PEFT config (peft_type="DENSE")
- DenseModel            — BaseTuner that replaces Linear → DenseLinear
- DensePeftModelForCausalLM — saves as merged full model
- get_dense_model()     — convenience constructor

Compatibility: peft==0.17.1, transformers==4.56.0, torch==2.7.1
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from peft.config import PeftConfig
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_adapters_to_merge,
    check_target_module_exists,
)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

DENSE_PEFT_TYPE = "DENSE"


# =========================================================================== #
# DenseVariant — abstract extension point (mirrors LoraVariant)               #
# =========================================================================== #

class DenseVariant:
    """Extension point for variants that transform a dense adapter update.

    Subclasses override the static methods to modify how ΔW is applied.
    The default (no variant) behaviour is a plain addition:  y = W₀x + ΔW x.
    """

    @staticmethod
    def init(module: "DenseLayer", adapter_name: str, **kwargs) -> None:
        raise NotImplementedError

    @staticmethod
    def get_delta_weight(
        module: "DenseLayer", adapter_name: str, delta: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def forward(
        module: "DenseLayer",
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError


# =========================================================================== #
# DenseLayer — full-rank adapter layer                                        #
# =========================================================================== #

class DenseLayer(BaseTunerLayer):
    """Base layer for full-rank additive adapters.

    Holds ``dense_delta`` (a ParameterDict of ΔW matrices) and an optional
    ``dense_variant`` dict that transforms the update.
    """

    adapter_layer_names = ("dense_delta",)
    other_param_names = ()

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        # ------------------------------------------------------------------ #
        # IMPORTANT (peft 0.17.1):  BaseTunerLayer declares class-level       #
        # defaults (_disable_adapters, _active_adapter, merged_adapters).     #
        # We must set *instance* attributes here so they don't leak across    #
        # different DenseLayer instances.                                     #
        # ------------------------------------------------------------------ #
        self.base_layer = base_layer
        self.dense_delta = nn.ParameterDict({})
        self.dense_variant: dict[str, DenseVariant] = {}
        self._disable_adapters = False
        self._active_adapter = "default"
        self.merged_adapters: list[str] = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if not isinstance(base_layer, nn.Linear):
            raise ValueError(
                f"Unsupported layer type {type(base_layer)}. "
                "Dense supports torch.nn.Linear only."
            )
        self.in_features: int = base_layer.in_features
        self.out_features: int = base_layer.out_features
        base_layer.requires_grad_(False)

    # -- adapter lifecycle ---------------------------------------------------

    def update_layer(
        self,
        adapter_name: str,
        inference_mode: bool = False,
        **kwargs,
    ) -> None:
        base_weight = self.get_base_layer().weight

        # A ZeRO-3 parameter is commonly exposed as an empty local tensor with
        # shape (0,), while its logical shape lives in ds_shape.  Never derive
        # adapter dimensions from the local partition: doing so permanently
        # creates a zero-sized parameter whose ds_shape is also (0,).
        logical_shape = (self.out_features, self.in_features)
        ds_shape = getattr(base_weight, "ds_shape", None)
        if ds_shape is not None and tuple(ds_shape) != logical_shape:
            raise ValueError(
                "Dense base-weight metadata is inconsistent: "
                f"Linear expects {logical_shape}, ZeRO reports {tuple(ds_shape)}."
            )

        if base_weight.device.type == "meta":
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            device = base_weight.device

        delta = nn.Parameter(
            torch.zeros(logical_shape, dtype=base_weight.dtype, device=device),
            requires_grad=True,
        )

        self.dense_delta[adapter_name] = delta
        if base_weight.device.type != "meta":
            self._move_adapter_to_device_of_base_layer(adapter_name)
        # peft 0.17.1: set_adapter() takes ONLY adapter_names (no inference_mode kwarg)
        self.set_adapter(self.active_adapters)
        # inference_mode is handled by BaseTuner.inject_adapter() after injection

    # -- delta weight (with optional variant projection) ---------------------

    def get_delta_weight(
        self, adapter_name: str, delta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return (possibly variant-transformed) ΔW for *adapter_name*."""
        delta = self.dense_delta[adapter_name] if delta is None else delta
        variant = self.dense_variant.get(adapter_name)
        if variant is not None:
            return variant.get_delta_weight(self, adapter_name, delta)
        return delta

    # -- merge / unmerge -----------------------------------------------------

    def merge(
        self,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter not in self.dense_delta:
                continue

            base_layer = self.get_base_layer()
            delta = self.get_delta_weight(active_adapter)
            if safe_merge:
                orig = base_layer.weight.data.clone() + delta.to(base_layer.weight.dtype)
                if not torch.isfinite(orig).all():
                    raise ValueError(
                        "NaNs or Infs detected while merging Dense adapter "
                        f"{active_adapter!r}."
                    )
                base_layer.weight.data = orig
            else:
                base_layer.weight.data += delta.to(base_layer.weight.dtype)

            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while self.merged_adapters:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self.dense_delta:
                continue
            weight = self.get_base_layer().weight
            weight.data -= self.get_delta_weight(active_adapter).to(weight.dtype)

    # -- forward -------------------------------------------------------------

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        adapter_names = kwargs.pop("adapter_names", None)
        if adapter_names is not None:
            raise NotImplementedError(
                "Dense adapters do not support per-example adapter_names."
            )

        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.dense_delta:
                    continue
                if active_adapter in self.dense_variant:
                    result = self.dense_variant[active_adapter].forward(
                        self,
                        active_adapter=active_adapter,
                        x=x,
                        result=result,
                        **kwargs,
                    )
                else:
                    delta = self.dense_delta[active_adapter]
                    result = result + F.linear(
                        x.to(delta.dtype), delta
                    ).to(result_dtype)
            result = result.to(result_dtype)

        return result.to(previous_dtype)

    def __repr__(self) -> str:
        return "dense." + super().__repr__()


# =========================================================================== #
# DenseLinear — concrete module that replaces nn.Linear                       #
# =========================================================================== #

class DenseLinear(nn.Module, DenseLayer):
    """A ``nn.Linear`` replacement holding a full-rank Dense adapter.

    Uses multiple inheritance: ``nn.Module`` provides parameter registration
    and device management; ``DenseLayer`` provides the adapter logic.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        adapter_name: str,
        inference_mode: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()  # nn.Module.__init__
        DenseLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, inference_mode=inference_mode)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return DenseLayer.forward(self, x, *args, **kwargs)

    def __repr__(self) -> str:
        return "dense." + nn.Module.__repr__(self)


# =========================================================================== #
# DenseConfig — PEFT configuration dataclass                                  #
# =========================================================================== #

@dataclass
class DenseConfig(PeftConfig):
    """Configuration for a full-rank additive weight adapter (peft_type="DENSE")."""

    target_modules: Optional[Union[list[str], str]] = field(default=None)
    exclude_modules: Optional[Union[list[str], str]] = field(default=None)
    layers_to_transform: Optional[Union[list[int], int]] = field(default=None)
    layers_pattern: Optional[Union[list[str], str]] = field(default=None)
    bias: str = field(default="none")
    modules_to_save: Optional[list[str]] = field(default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.peft_type = DENSE_PEFT_TYPE

        if self.bias != "none":
            raise ValueError("Dense adapters do not support training bias parameters.")
        if self.modules_to_save:
            raise ValueError("Dense adapters do not support modules_to_save.")

        self.target_modules = (
            set(self.target_modules)
            if isinstance(self.target_modules, list)
            else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules)
            if isinstance(self.exclude_modules, list)
            else self.exclude_modules
        )

        if (
            isinstance(self.target_modules, str)
            and self.layers_to_transform is not None
        ):
            raise ValueError(
                "layers_to_transform cannot be used when target_modules is a regex string."
            )
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError(
                "layers_pattern cannot be used when target_modules is a regex string."
            )
        if self.layers_pattern and self.layers_to_transform is None:
            raise ValueError(
                "layers_pattern requires layers_to_transform to be specified."
            )


# =========================================================================== #
# DenseModel — BaseTuner that replaces Linear → DenseLinear                   #
# =========================================================================== #

class DenseModel(BaseTuner):
    """PEFT tuner that replaces selected ``nn.Linear`` layers with ``DenseLinear``.

    Implements all 6 ``@abstractmethod`` s and 3 essential non-abstract methods
    required by ``BaseTuner`` in peft 0.17.1.  See the 9-item checklist in
    ``docs/superpowers/plans/2026-07-26-curvature-dense-warning.md``.
    """

    prefix: str = "dense_"
    tuner_layer_cls = DenseLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    # ================================================================== #
    # 6 abstractmethod overrides                                         #
    # ================================================================== #

    # -- 1. _prepare_adapter_config -------------------------------------------

    @staticmethod
    def _prepare_adapter_config(peft_config: PeftConfig, model_config: dict):
        if peft_config.target_modules is None:
            target_modules = DenseModel.target_module_mapping.get(
                model_config.get("model_type", ""), []
            )
            if target_modules:
                peft_config.target_modules = (
                    set(target_modules)
                    if not isinstance(target_modules, str)
                    else target_modules
                )
            else:
                raise ValueError(
                    "Please specify `target_modules` in `peft_config`."
                )
        return peft_config

    # -- 2. _check_target_module_exists --------------------------------------

    @staticmethod
    def _check_target_module_exists(peft_config: PeftConfig, key: str) -> bool:
        return check_target_module_exists(peft_config, key)

    # -- 3. _create_and_replace -----------------------------------------------
    # peft 0.17.1: inject_adapter calls this with `current_key=key` only.
    # inference_mode is handled by inject_adapter internally — do NOT pass
    # it down to update_layer / _create_new_module.

    def _create_and_replace(
        self,
        dense_config: DenseConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        **optional_kwargs,
    ) -> None:
        if current_key is None:
            raise ValueError("Current key cannot be None.")

        if isinstance(target, DenseLayer):
            target.update_layer(adapter_name)
            return

        new_module = self._create_new_module(adapter_name, target)
        if adapter_name not in self.active_adapters:
            new_module.requires_grad_(False)
        self._replace_module(parent, target_name, new_module, target)

    # -- 4. _mark_only_adapters_as_trainable ----------------------------------

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if "dense_delta" not in n:
                p.requires_grad = False
        for n, p in model.named_parameters():
            if "dense_delta" in n:
                p.requires_grad = True

    # -- 5 & 6. disable / enable adapter layers -------------------------------
    # Use enable_adapters(bool) — disable_adapters is a read-only property.

    def disable_adapter_layers(self) -> None:
        for module in self.modules():
            if isinstance(module, BaseTunerLayer):
                module.enable_adapters(enabled=False)

    def enable_adapter_layers(self) -> None:
        for module in self.modules():
            if isinstance(module, BaseTunerLayer):
                module.enable_adapters(enabled=True)

    # ================================================================== #
    # 3 non-abstract but essential methods                               #
    # ================================================================== #

    # -- 7. set_adapter (model-level) -----------------------------------------
    # Called by inject_adapter itself right after injection completes.

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        for module in self.model.modules():
            if isinstance(module, DenseLayer):
                if module.merged:
                    warnings.warn(
                        "Adapter cannot be set when the model is merged. "
                        "Unmerging the model first."
                    )
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    # -- 8. _replace_module ---------------------------------------------------
    # NOT inherited from BaseTuner in 0.17.1; each tuner defines its own.

    def _replace_module(self, parent, child_name, new_module, child) -> None:
        setattr(parent, child_name, new_module)

        if hasattr(child, "base_layer"):
            child = child.base_layer

        weight = (
            child.weight
            if hasattr(child, "weight")
            else next(child.parameters())
        )

        meta = torch.device("meta")
        if not any(p.device == meta for p in new_module.parameters()):
            new_module.to(weight.device)

    # -- 9. __getattr__ -------------------------------------------------------
    # Forward unknown attributes to the wrapped model (config, generate, etc.).

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model":
                raise
            return getattr(self.model, name)

    # ================================================================== #
    # helpers                                                             #
    # ================================================================== #

    @staticmethod
    def _create_new_module(
        adapter_name: str,
        target: nn.Module,
    ) -> DenseLinear:
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if not isinstance(target_base_layer, nn.Linear):
            raise ValueError(
                f"Target module {type(target)} is not supported. "
                "Dense supports torch.nn.Linear only."
            )
        # Guard against mixing Dense with another PEFT tuner on the same layer.
        if isinstance(target, BaseTunerLayer) and not isinstance(target, DenseLayer):
            raise ValueError(
                "Dense cannot be mixed with another PEFT tuner layer on the same module."
            )

        return DenseLinear(target, adapter_name)

    def _cast_adapter_dtype(
        self, adapter_name: str, autocast_adapter_dtype: bool = True
    ) -> None:
        # Dense ΔW intentionally follows the base weight dtype — no-op.
        return None


# =========================================================================== #
# DensePeftModelForCausalLM — saves a merged full model                       #
# =========================================================================== #

class DensePeftModelForCausalLM:
    """Mixin that overrides ``save_pretrained`` to export a merged full model.

    Dense ΔW has the same shape as base weights, so saving just the adapter
    provides no storage benefit.  Instead we merge ΔW into base weights and
    save as a standard HuggingFace model (no PEFT required to reload).
    """

    @staticmethod
    def _iter_dense_modules(model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, DenseLinear):
                yield name, module

    @staticmethod
    def _find_root_prefix(state_dict: dict[str, torch.Tensor]) -> str:
        """Discover the root prefix (e.g. "model.") from state keys.

        DenseModel wraps a HuggingFace model as ``self.model``, so all
        state-dict keys are prefixed with ``"model."`` (the DenseModel's
        attribute name).  The HF model itself may also use ``"model."``
        as its own internal prefix (e.g. ``model.model.embed_tokens.weight``).

        We must strip ONLY the outermost DenseModel prefix (``"model."``)
        and preserve the HF model's internal naming (``model.embed_tokens...``)
        so that vLLM / HuggingFace can reload the weights correctly.
        """
        # The DenseModel attribute holding the HF model is always "model".
        # Every key in DenseModel.state_dict() therefore starts with "model.".
        # Simply return that fixed prefix.
        return "model."

    @classmethod
    def _merged_base_state_dict(
        cls,
        model: nn.Module,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        source = model.state_dict() if state_dict is None else state_dict
        root = cls._find_root_prefix(source)

        merged: dict[str, torch.Tensor] = {}
        for key, value in source.items():
            if root and not key.startswith(root):
                continue
            plain_key = key[len(root):] if root else key
            # Skip adapter-only keys
            if (
                ".dense_delta." in plain_key
                or ".U_in_bar_" in plain_key
                or ".U_out_bar_" in plain_key
            ):
                continue
            merged[plain_key.replace(".base_layer.", ".")] = value

        # Merge ΔW into base weights
        for _name, module in cls._iter_dense_modules(model):
            # _name includes the root prefix (e.g. "model.linear").
            # Strip root to get the key we'll store in the merged dict.
            rel_name = _name[len(root):] if root and _name.startswith(root) else _name
            # Reconstruct the state-dict key for this module's base weight
            for key in source:
                if key.endswith(f"{_name}.base_layer.weight"):
                    weight = source[key].clone()
                    if not module.merged:
                        for adapter_name in module.active_adapters:
                            delta_key = f"{_name}.dense_delta.{adapter_name}"
                            delta = source.get(delta_key)
                            if delta is None:
                                delta = module.dense_delta[adapter_name]
                            weight += module.get_delta_weight(
                                adapter_name, delta
                            ).to(weight.dtype)
                    merged[f"{rel_name}.weight"] = weight
                    break

        return merged

    @staticmethod
    def save_pretrained_dense(
        model: nn.Module,
        save_directory: str,
        safe_serialization: bool = True,
        is_main_process: bool = True,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> None:
        """Save the model with ΔW merged into base weights."""
        merged = DensePeftModelForCausalLM._merged_base_state_dict(model, state_dict)
        model.save_pretrained(
            save_directory,
            state_dict=merged,
            safe_serialization=safe_serialization,
            is_main_process=is_main_process,
            **kwargs,
        )


# =========================================================================== #
# Registration & convenience constructor                                      #
# =========================================================================== #

def register_dense_peft_method() -> None:
    """Register the Dense tuner in PEFT's dispatch maps (idempotent)."""
    from peft.mapping import (
        PEFT_TYPE_TO_CONFIG_MAPPING,
        PEFT_TYPE_TO_PREFIX_MAPPING,
        PEFT_TYPE_TO_TUNER_MAPPING,
    )

    # Always overwrite — the same class may arrive via different import
    # paths (e.g. `dense_peft.DenseConfig` vs `utils.dense_peft.DenseConfig`)
    # and `is`-based identity checks will fail.  Overwriting with the
    # same logical value is harmless and makes registration truly idempotent.
    PEFT_TYPE_TO_CONFIG_MAPPING[DENSE_PEFT_TYPE] = DenseConfig
    PEFT_TYPE_TO_TUNER_MAPPING[DENSE_PEFT_TYPE] = DenseModel
    PEFT_TYPE_TO_PREFIX_MAPPING[DENSE_PEFT_TYPE] = DenseModel.prefix

def get_dense_model(
    model: nn.Module,
    config: DenseConfig,
    adapter_name: str = "default",
    low_cpu_mem_usage: bool = False,
) -> DenseModel:
    """Wrap *model* with a Dense adapter and return the DenseModel tuner.

    After this call the original ``nn.Linear`` layers targeted by
    *config.target_modules* are replaced with ``DenseLinear`` instances.
    """
    register_dense_peft_method()
    return DenseModel(
        model,
        config,
        adapter_name=adapter_name,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )


# Eager registration so "DENSE" is recognised on import.
register_dense_peft_method()


__all__ = [
    "DENSE_PEFT_TYPE",
    "DenseConfig",
    "DenseLayer",
    "DenseLinear",
    "DenseModel",
    "DensePeftModelForCausalLM",
    "DenseVariant",
    "get_dense_model",
    "register_dense_peft_method",
]
