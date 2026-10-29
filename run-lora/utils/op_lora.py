"""Orthogonal Projection LoRA (OPLoRA).

Constrains the LoRA update to the bilateral orthogonal complement of the
top-k singular subspaces of the frozen base weight W0:

    ΔW = (α/r) P_L B A P_R
    P_L = I - U_k U_k^T,   P_R = I - V_k V_k^T

so the top-k singular triplets of W0 are exactly preserved. After training,
projections are folded into A/B so the saved adapter is standard LoRA.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import TrainerCallback

from peft.tuners.lora.layer import LoraLayer, LoraVariant

logger = logging.getLogger(__name__)


def compute_top_k_bases(
    weight: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return top-k left/right singular vectors and values of ``weight``.

    SVD is computed in float32. ``k`` is clamped to ``min(out, in)``.
    Returns ``(U_k, S_k, V_k)`` with shapes ``(out, k_eff)``, ``(k_eff,)``,
    ``(in, k_eff)``.
    """
    if k <= 0:
        raise ValueError(f"OPLoRA k must be a positive integer, got {k}.")
    weight_f = weight.detach().float()
    u, s, vh = torch.linalg.svd(weight_f, full_matrices=False)
    k_eff = min(int(k), int(s.numel()))
    u_k = u[:, :k_eff].contiguous()
    s_k = s[:k_eff].contiguous()
    v_k = vh[:k_eff, :].T.contiguous()
    return u_k, s_k, v_k


def project_delta(
    weight_b: torch.Tensor,
    weight_a: torch.Tensor,
    u_k: torch.Tensor,
    v_k: torch.Tensor,
    scaling: float = 1.0,
) -> torch.Tensor:
    """Return ``scaling * P_L B A P_R`` without materializing the dense projectors."""
    a_eff = weight_a - (weight_a @ v_k) @ v_k.T
    b_eff = weight_b - u_k @ (u_k.T @ weight_b)
    return scaling * (b_eff @ a_eff)


def subspace_alignment(delta: torch.Tensor, u_k: torch.Tensor) -> torch.Tensor:
    """Diagnostic ρ_k = ||Q_k ΔW||_F^2 / ||ΔW||_F^2, Q_k = U_k U_k^T."""
    denom = torch.linalg.vector_norm(delta).square()
    if denom.item() == 0:
        return delta.new_zeros(())
    q_delta = u_k @ (u_k.T @ delta)
    return torch.linalg.vector_norm(q_delta).square() / denom


def _get_active_adapter(module) -> Optional[str]:
    if hasattr(module, "active_adapters") and module.active_adapters:
        return module.active_adapters[0]
    if hasattr(module, "_active_adapter"):
        return module._active_adapter
    return None


def _base_weight(module: LoraLayer) -> torch.Tensor:
    return module.base_layer.weight if hasattr(module, "base_layer") else module.weight


class OPLora(LoraVariant):
    """LoRA variant that projects updates onto the complement of W0's top-k SVD bases."""

    @staticmethod
    def init(module: LoraLayer, adapter_name: str, k: int = 16, **kwargs) -> None:
        base_weight = _base_weight(module)
        u_k, _, v_k = compute_top_k_bases(base_weight, k)
        module.register_buffer(
            f"U_k_{adapter_name}",
            u_k.to(device=base_weight.device, dtype=base_weight.dtype),
        )
        module.register_buffer(
            f"V_k_{adapter_name}",
            v_k.to(device=base_weight.device, dtype=base_weight.dtype),
        )

    @staticmethod
    def _projection_buffers(module: LoraLayer, active_adapter: str):
        u_k = getattr(module, f"U_k_{active_adapter}", None)
        v_k = getattr(module, f"V_k_{active_adapter}", None)
        if u_k is None or v_k is None:
            raise RuntimeError(
                f"OPLoRA adapter {active_adapter!r} is active, but SVD buffers are missing."
            )
        return u_k, v_k

    @staticmethod
    def forward(
        module: LoraLayer,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        u_k, v_k = OPLora._projection_buffers(module, active_adapter)
        lora_a = module.lora_A[active_adapter]
        lora_b = module.lora_B[active_adapter]
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]
        weight_a = lora_a.weight
        weight_b = lora_b.weight

        compute_dtype = weight_a.dtype
        v_k = v_k.to(dtype=compute_dtype)
        u_k = u_k.to(dtype=compute_dtype)

        a_eff = weight_a - (weight_a @ v_k) @ v_k.T
        b_eff = weight_b - u_k @ (u_k.T @ weight_b)

        x_in = dropout(x)
        if x_in.dtype != compute_dtype:
            x_in = x_in.to(compute_dtype)
        h = F.linear(F.linear(x_in, a_eff), b_eff)
        return result + (h * scaling).to(result.dtype)

    @staticmethod
    def _compute_delta_weight(module: LoraLayer, active_adapter: str) -> torch.Tensor:
        u_k, v_k = OPLora._projection_buffers(module, active_adapter)
        weight_a = module.lora_A[active_adapter].weight
        weight_b = module.lora_B[active_adapter].weight
        scaling = module.scaling[active_adapter]
        dtype = weight_b.dtype

        cast_to_fp32 = weight_b.device.type == "cpu" and dtype in (torch.float16, torch.bfloat16)
        if cast_to_fp32:
            weight_a = weight_a.float()
            weight_b = weight_b.float()
            u_k = u_k.float()
            v_k = v_k.float()
        else:
            u_k = u_k.to(dtype=dtype)
            v_k = v_k.to(dtype=dtype)

        delta = project_delta(weight_b, weight_a, u_k, v_k, scaling)
        return delta.to(dtype=dtype) if cast_to_fp32 else delta

    @staticmethod
    def merge_safe(module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        delta = OPLora._compute_delta_weight(module, active_adapter)
        return orig_weight + delta.to(orig_weight.dtype)

    @staticmethod
    def merge_unsafe(module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor) -> None:
        delta = OPLora._compute_delta_weight(module, active_adapter)
        orig_weight.data += delta.to(orig_weight.dtype)

    @staticmethod
    def unmerge(module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        delta = OPLora._compute_delta_weight(module, active_adapter)
        return orig_weight - delta.to(orig_weight.dtype)


def apply_op_lora(model, k: int = 16, adapter_name: Optional[str] = None):
    from peft.tuners.lora.layer import Linear as PeftLoraLinear

    augmented = 0
    total_lora = 0
    for _, module in model.named_modules():
        if not isinstance(module, PeftLoraLinear):
            continue
        total_lora += 1
        active = adapter_name or _get_active_adapter(module)
        if active is None:
            raise ValueError("OPLoRA could not determine the active PEFT adapter name.")
        if active in getattr(module, "lora_variant", {}):
            raise ValueError(
                f"Adapter {active!r} already has a LoRA variant attached; OPLoRA expects plain LoRA."
            )
        module.lora_variant[active] = OPLora()
        OPLora.init(module, active, k=k)
        augmented += 1

    if augmented == 0:
        raise ValueError("OPLoRA found no PEFT LoRA Linear modules to augment.")
    logger.info("OPLoRA: augmented %s/%s LoRA layers with k=%s", augmented, total_lora, k)
    return model


def fold_oplora_projections(model, adapter_name: Optional[str] = None) -> int:
    """Bake P_L / P_R into A/B and strip OPLoRA buffers so the adapter is plain LoRA."""
    from peft.tuners.lora.layer import Linear as PeftLoraLinear

    folded = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, PeftLoraLinear):
                continue
            active = adapter_name or _get_active_adapter(module)
            if active is None:
                continue
            if not isinstance(getattr(module, "lora_variant", {}).get(active), OPLora):
                continue
            u_k, v_k = OPLora._projection_buffers(module, active)
            weight_a = module.lora_A[active].weight
            weight_b = module.lora_B[active].weight
            v_cast = v_k.to(dtype=weight_a.dtype, device=weight_a.device)
            u_cast = u_k.to(dtype=weight_b.dtype, device=weight_b.device)
            if v_cast.shape[1] == 0 and u_cast.shape[1] == 0:
                raise RuntimeError(f"Refusing to fold empty OPLoRA projections for module {name}.")
            if v_cast.shape[1] > 0:
                weight_a.copy_(weight_a - (weight_a @ v_cast) @ v_cast.T)
            if u_cast.shape[1] > 0:
                weight_b.copy_(weight_b - u_cast @ (u_cast.T @ weight_b))
            module.lora_variant.pop(active, None)
            module._buffers.pop(f"U_k_{active}", None)
            module._buffers.pop(f"V_k_{active}", None)
            folded += 1
    logger.info("OPLoRA: folded %d LoRA layers into plain adapter weights.", folded)
    return folded


class OPLoRAFoldCallback(TrainerCallback):
    """Fold OPLoRA projections before SavePeftModelCallback writes the adapter."""

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        fold_oplora_projections(model)
