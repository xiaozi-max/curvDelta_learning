"""CurvatureDense — full-rank adapter with KFAC bilateral curvature projection.

Provides:
- CurvatureDense(DenseVariant)       — bilateral projection variant
- apply_curvature_dense(model)       — attaches CurvatureDense to DenseLinear layers
- get_curvature_dense_model(model)   — convenience: DenseModel + CurvatureDense
- CurvatureDenseTrainerMixin         — KFAC collection, projection, folding
- CurvatureDenseTrainer              — concrete Trainer class

Reuses KFACTracker, compute_projection_cache, CurvatureTextDataset,
CurvatureDataCollator from curvature_lora.py.

Compatibility: peft==0.17.1, transformers==4.56.0, torch==2.7.1
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import Trainer, TrainerCallback

# Ensure the parent run-lora/ directory is on sys.path so that
# `from utils.xxx` resolves correctly regardless of whether this file
# is imported from train.py (run-lora/) or from a test (run-lora/utils/).
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from utils.curvature_lora import (
    FISHER_SAMPLE_SEED,
    FISHER_TEMPERATURE,
    CurvatureDataCollator,
    CurvatureTextDataset,
    KFACTracker,
    compute_projection_cache,
)
from utils.dense_peft import (
    DenseConfig,
    DenseLayer,
    DenseLinear,
    DensePeftModelForCausalLM,
    DenseVariant,
    get_dense_model,
)


logger = logging.getLogger(__name__)


# =========================================================================== #
# _DisabledDenseAdapterModel — disables Dense adapters for ref-model forward   #
# =========================================================================== #
# curvature_lora's _DisabledAdapterModel uses model.disable_adapter() which is
# a PeftModel method NOT available on raw DenseModel.  We provide a Dense-
# specific equivalent using disable_adapter_layers() / enable_adapter_layers().

class _DisabledDenseAdapterModel:
    """Wraps a DenseModel so that __call__ runs with adapters disabled."""

    def __init__(self, model):
        object.__setattr__(self, "_model", model)

    def __call__(self, *args, **kwargs):
        was_training = self._model.training
        self._model.eval()
        try:
            self._model.disable_adapter_layers()
            return self._model(*args, **kwargs)
        finally:
            self._model.enable_adapter_layers()
            if was_training:
                self._model.train()

    def __getattr__(self, name):
        return getattr(self._model, name)


# =========================================================================== #
# CurvatureDense — DenseVariant that applies bilateral projection             #
# =========================================================================== #

class CurvatureDense(DenseVariant):
    """Dense variant with bilateral low-curvature projection.

    Forward computes::

        update = (I - Ū_out Ū_out^T) @ ΔW @ (I - Ū_in Ū_in^T) @ x

    where Ū_in and Ū_out are frozen high-curvature KFAC eigenvector bases
    stored as non-persistent buffers.

    Uses **token-level** projection (not parameter-level folding) because
    for a full-rank ΔW, the O(d_out·d_in·k) folding cost dwarfs the
    O(B·T·(d_in+d_out)·k) token-level cost at typical batch sizes.
    """

    @staticmethod
    def init(module: DenseLayer, adapter_name: str, **kwargs) -> None:
        base_weight = module.get_base_layer().weight
        module.register_buffer(
            f"U_in_bar_{adapter_name}",
            base_weight.new_zeros((module.in_features, 0)),
            persistent=False,
        )
        module.register_buffer(
            f"U_out_bar_{adapter_name}",
            base_weight.new_zeros((module.out_features, 0)),
            persistent=False,
        )

    @staticmethod
    def get_delta_weight(
        module: DenseLayer, adapter_name: str, delta: torch.Tensor
    ) -> torch.Tensor:
        U_in_bar = getattr(module, f"U_in_bar_{adapter_name}").to(delta)
        U_out_bar = getattr(module, f"U_out_bar_{adapter_name}").to(delta)

        projected = delta
        if U_out_bar.shape[1] > 0:
            projected = projected - U_out_bar @ (U_out_bar.T @ projected)
        if U_in_bar.shape[1] > 0:
            projected = projected - (projected @ U_in_bar) @ U_in_bar.T
        return projected

    @staticmethod
    def forward(
        module: DenseLayer,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        delta = module.dense_delta[active_adapter]
        U_in_bar = getattr(module, f"U_in_bar_{active_adapter}").to(delta)
        U_out_bar = getattr(module, f"U_out_bar_{active_adapter}").to(delta)

        # Token-level bilateral projection
        x_projected = x.to(delta.dtype)
        if U_in_bar.shape[1] > 0:
            x_projected = x_projected - (x_projected @ U_in_bar) @ U_in_bar.T

        update = F.linear(x_projected, delta)
        if U_out_bar.shape[1] > 0:
            update = update - (update @ U_out_bar) @ U_out_bar.T

        return result + update.to(result.dtype)


# =========================================================================== #
# apply_curvature_dense / get_curvature_dense_model                           #
# =========================================================================== #

def _get_active_adapter(module) -> Optional[str]:
    if hasattr(module, "active_adapters") and module.active_adapters:
        return module.active_adapters[0]
    if hasattr(module, "_active_adapter"):
        return module._active_adapter
    return None


def apply_curvature_dense(model, adapter_name: Optional[str] = None):
    """Attach CurvatureDense variant to every DenseLinear in *model*."""
    augmented = 0
    total_dense = 0
    for _, module in model.named_modules():
        if not isinstance(module, DenseLinear):
            continue
        total_dense += 1
        active = adapter_name or _get_active_adapter(module)
        if active is None or active not in module.dense_delta:
            continue
        if active in module.dense_variant:
            raise ValueError(
                f"Adapter {active!r} already has a Dense variant attached."
            )
        module.dense_variant[active] = CurvatureDense()
        CurvatureDense.init(module, active)
        augmented += 1

    if augmented == 0:
        raise ValueError(
            "CurvatureDense found no DenseLinear modules to augment."
        )
    logger.info(
        "CurvatureDense: augmented %s/%s Dense layers", augmented, total_dense
    )
    return model


def get_curvature_dense_model(
    model,
    config: DenseConfig,
    adapter_name: str = "default",
    low_cpu_mem_usage: bool = False,
):
    """Construct DenseModel then attach CurvatureDense variant."""
    peft_model = get_dense_model(
        model,
        config,
        adapter_name=adapter_name,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    return apply_curvature_dense(peft_model, adapter_name=adapter_name)


# =========================================================================== #
# CurvatureDenseCallback — hooks into Trainer lifecycle                       #
# =========================================================================== #

class CurvatureDenseCallback(TrainerCallback):
    """Triggers KFAC collection at train_begin and folding at train_end.

    No ``on_optimizer_step`` hook — the trust-region trick is deprecated.
    """

    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer.curvature_on_train_begin()

    def on_train_end(self, args, state, control, **kwargs):
        self.trainer.curvature_on_train_end()


# =========================================================================== #
# CurvatureDenseTrainerMixin — KFAC + projection + folding                    #
# =========================================================================== #

class CurvatureDenseTrainerMixin:
    """Mixin that manages KFAC collection and projection for CurvatureDense.

    Reuses the KFACTracker / compute_projection_cache machinery from
    curvature_lora.py.  The Dense-specific parts are: discovering
    DenseLinear modules, writing projection buffers via dense_delta
    reference, and folding projections into dense_delta at train end.
    """

    def __init__(
        self,
        *args,
        curvature_data_path: str,
        curvature_energy_ratio: float = 0.9,
        curvature_collection_batch_size: int = 1,
        curvature_num_chunks: int = 1,
        curvature_alpha_tr: float = 0.0,      # deprecated — ignored
        curvature_delta_tr: float = 1e-12,    # deprecated — ignored
        **kwargs,
    ):
        # -- validation ------------------------------------------------------ #
        if not curvature_data_path:
            raise ValueError(
                "curvature_data_path is required for CurvatureDense."
            )
        if not 0 < float(curvature_energy_ratio) <= 1:
            raise ValueError(
                f"curvature_energy_ratio must be in (0, 1], "
                f"got {curvature_energy_ratio}."
            )
        if int(curvature_collection_batch_size) < 1:
            raise ValueError(
                f"curvature_collection_batch_size must be >= 1, "
                f"got {curvature_collection_batch_size}."
            )
        if int(curvature_num_chunks) < 1:
            raise ValueError(
                f"curvature_num_chunks must be >= 1, "
                f"got {curvature_num_chunks}."
            )

        # -- store config ---------------------------------------------------- #
        self.curvature_data_path = curvature_data_path
        self.curvature_energy_ratio = float(curvature_energy_ratio)
        self.curvature_collection_batch_size = int(curvature_collection_batch_size)
        self.curvature_num_chunks = int(curvature_num_chunks)
        # trust-region is deprecated — hardcode to disabled
        self.curvature_alpha_tr = 0.0
        self.curvature_delta_tr = 1e-12

        # -- internal state -------------------------------------------------- #
        self._target_modules: Dict[str, torch.nn.Module] = {}
        self._adapter_names: Dict[str, str] = {}
        self._local_projections: Dict[str, tuple] = {}
        self._curvature_fisher_generator: Optional[torch.Generator] = None
        self.ref_model = None

        # -- chain to parent Trainer ----------------------------------------- #
        super().__init__(*args, **kwargs)

        # DenseModel.forward has *args/**kwargs (inherited from BaseTuner),
        # so the Trainer's _remove_unused_columns can't match dataset columns
        # to the forward signature.  Disable column filtering.
        self.args.remove_unused_columns = False

        # -- setup ----------------------------------------------------------- #
        self._fail_on_zero3()
        self.add_callback(CurvatureDenseCallback(self))
        self._setup_curvature()

    # -- DeepSpeed guard -----------------------------------------------------

    def _fail_on_zero3(self) -> None:
        if not getattr(self, "is_deepspeed_enabled", False):
            return
        plugin = getattr(getattr(self, "accelerator", None), "state", None)
        plugin = getattr(plugin, "deepspeed_plugin", None)
        config = getattr(plugin, "deepspeed_config", None)
        if not isinstance(config, dict):
            raise RuntimeError(
                "CurvatureDense could not inspect DeepSpeed config."
            )
        stage = config.get("zero_optimization", {}).get("stage")
        if str(stage) == "3":
            raise ValueError(
                "CurvatureDense supports DeepSpeed ZeRO-2 only; "
                "ZeRO-3 is intentionally unsupported."
            )

    # -- module discovery ----------------------------------------------------

    def _setup_curvature(self) -> None:
        """Discover DenseLinear modules with attached CurvatureDense variants."""
        for name, module in self.model.named_modules():
            if not isinstance(module, DenseLinear):
                continue
            if not getattr(module, "dense_variant", {}):
                continue
            self._target_modules[name] = module
            self._adapter_names[name] = next(iter(module.dense_variant))

        if not self._target_modules:
            raise ValueError(
                "CurvatureDense trainer found no modules with attached "
                "curvature variants."
            )

        # Round-robin assignment for distributed KFAC collection
        module_names = list(self._target_modules.keys())
        rank = self.args.process_index
        world_size = self.args.world_size
        assigned_modules = {
            name: self._target_modules[name]
            for i, name in enumerate(module_names)
            if i % world_size == rank
        }

        self._tracker = KFACTracker(
            assigned_modules,
            device=self.accelerator.device,
            num_chunks=self.curvature_num_chunks,
        )
        self._curvature_loader = self._build_curvature_loader()

    # -- data loading (shared with curvature_lora) ---------------------------

    def _tokenizer(self):
        tokenizer = (
            getattr(self, "processing_class", None)
            or getattr(self, "tokenizer", None)
        )
        if tokenizer is None:
            raise ValueError(
                "CurvatureDense trainer requires a tokenizer/processing_class."
            )
        return tokenizer

    def _build_curvature_loader(self) -> DataLoader:
        tokenizer = self._tokenizer()
        dataset = CurvatureTextDataset(
            self.curvature_data_path, tokenizer, self.args.model_max_length
        )
        sampler = SequentialSampler(dataset)
        return DataLoader(
            dataset,
            batch_size=self.curvature_collection_batch_size,
            sampler=sampler,
            collate_fn=CurvatureDataCollator(tokenizer),
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _get_curvature_fisher_generator(
        self, device: torch.device
    ) -> torch.Generator:
        generator = self._curvature_fisher_generator
        if generator is None or generator.device != device:
            generator = torch.Generator(device=device)
            generator.manual_seed(FISHER_SAMPLE_SEED)
            self._curvature_fisher_generator = generator
        return generator

    def _valid_target_mask(self, inputs) -> torch.Tensor:
        if "labels" in inputs:
            return inputs["labels"][..., 1:].ne(-100)
        return inputs["attention_mask"][..., 1:].bool()

    def _compute_curvature_fisher_loss(self, model, inputs):
        """KL-style Fisher loss using reference model for target sampling."""
        if self.ref_model is None:
            self.ref_model = _DisabledDenseAdapterModel(model)
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)
            ref_logits = ref_outputs.logits[..., :-1, :] / FISHER_TEMPERATURE
            ref_probs = F.softmax(ref_logits, dim=-1)

        outputs = model(**inputs)
        student_logits = outputs.logits[..., :-1, :]
        valid_mask = self._valid_target_mask(inputs).to(student_logits.device)
        if valid_mask.sum().item() == 0:
            raise ValueError(
                "CurvatureDense Fisher collection found no valid next-token "
                "targets in a batch."
            )
        flat_ref_probs = ref_probs.to(student_logits.device)[valid_mask]
        generator = self._get_curvature_fisher_generator(student_logits.device)
        sampled_targets = torch.multinomial(
            flat_ref_probs.float(),
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(-1)
        flat_student_logits = student_logits[valid_mask]
        return F.cross_entropy(
            flat_student_logits, sampled_targets, reduction="mean"
        )

    # -- KFAC collection (factorised per chunk) ------------------------------

    def _collect_factors(self) -> None:
        """Collect KFAC A/S factors and compute projection bases.

        Factorised per chunk: each chunk's accumulators live on GPU only
        during its collection pass, keeping VRAM bounded.
        """
        tracker = self._tracker
        self._local_projections = {}
        is_main = self.accelerator.is_main_process

        for chunk_idx in range(tracker.num_chunks):
            self._curvature_fisher_generator = None
            tracker.activate_chunk(chunk_idx)
            tracker.clear_accumulators()
            tracker.install_hooks()
            self.model.eval()

            chunk_names = tracker.active_module_names()
            desc = (
                f"Collecting CurvatureDense KFAC factors "
                f"({chunk_idx + 1}/{tracker.num_chunks})"
                if tracker.num_chunks > 1
                else "Collecting CurvatureDense KFAC factors"
            )

            try:
                pbar = tqdm(
                    self._curvature_loader,
                    desc=desc,
                    disable=not is_main,
                    dynamic_ncols=True,
                )
                for batch in pbar:
                    batch = self._prepare_inputs(batch)
                    self.model.zero_grad(set_to_none=True)
                    with tracker.capture(batch["attention_mask"]):
                        loss = self._compute_curvature_fisher_loss(
                            self.model, batch
                        )
                        with self.accelerator.no_sync(self.model):
                            self.accelerator.backward(loss)
                    self.model.zero_grad(set_to_none=True)
                    pbar.set_postfix({"fisher_loss": f"{loss.item():.4f}"})
            finally:
                tracker.remove_hooks()
                self.model.train()

            # Normalise and compute projections per module in chunk
            for name in chunk_names:
                if tracker.kfac_n[name] == 0:
                    raise ValueError(
                        f"CurvatureDense collected no valid tokens "
                        f"for module {name}."
                    )
                tracker.kfac_A[name] /= tracker.kfac_n[name]
                tracker.kfac_S[name] /= tracker.kfac_n[name]

                U_in = compute_projection_cache(
                    tracker.kfac_A[name],
                    energy_ratio=self.curvature_energy_ratio,
                )
                U_out = compute_projection_cache(
                    tracker.kfac_S[name],
                    energy_ratio=self.curvature_energy_ratio,
                )
                if U_in.shape[1] == 0 and U_out.shape[1] == 0:
                    raise ValueError(
                        f"CurvatureDense produced empty projection bases "
                        f"for module {name}; KFAC factors have zero energy."
                    )
                self._local_projections[name] = (U_in.cpu(), U_out.cpu())

            tracker.deactivate_chunk(chunk_idx)

    # -- projection application ----------------------------------------------

    def _apply_projections(self) -> None:
        """Write projection buffers to DenseLinear modules."""
        try:
            if self.args.world_size > 1:
                self._distributed_apply_projections()
            else:
                for name, module in self._target_modules.items():
                    adapter_name = self._adapter_names[name]
                    U_in_bar, U_out_bar = self._local_projections[name]
                    ref = module.dense_delta[adapter_name]
                    module._buffers[f"U_in_bar_{adapter_name}"] = U_in_bar.to(ref)
                    module._buffers[f"U_out_bar_{adapter_name}"] = U_out_bar.to(ref)
        except Exception:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            raise

    def _distributed_apply_projections(self) -> None:
        """Broadcast projections so every rank holds identical buffers."""
        device = self.accelerator.device
        rank = self.args.process_index
        world_size = self.args.world_size
        module_names = list(self._target_modules.keys())

        # Phase 1: all-reduce projection shapes
        shapes = torch.zeros(len(module_names), 2, dtype=torch.long, device=device)
        for i, name in enumerate(module_names):
            if i % world_size == rank:
                U_in, U_out = self._local_projections[name]
                shapes[i, 0], shapes[i, 1] = U_in.shape[1], U_out.shape[1]
        shapes = self.accelerator.reduce(shapes, reduction="sum")

        # Phase 2: share projections, write buffers
        for i, name in enumerate(module_names):
            owner = i % world_size
            module = self._target_modules[name]
            adapter_name = self._adapter_names[name]
            ref = module.dense_delta[adapter_name]
            in_cols, out_cols = (
                int(shapes[i, 0].item()),
                int(shapes[i, 1].item()),
            )

            if owner == rank:
                buf_in = self._local_projections[name][0].to(
                    device=device, dtype=torch.float32
                )
            else:
                buf_in = torch.zeros(
                    module.in_features, in_cols,
                    dtype=torch.float32, device=device,
                )
            buf_in = self.accelerator.reduce(buf_in, reduction="sum").to(ref)
            module._buffers[f"U_in_bar_{adapter_name}"] = buf_in

            if owner == rank:
                buf_out = self._local_projections[name][1].to(
                    device=device, dtype=torch.float32
                )
            else:
                buf_out = torch.zeros(
                    module.out_features, out_cols,
                    dtype=torch.float32, device=device,
                )
            buf_out = self.accelerator.reduce(buf_out, reduction="sum").to(ref)
            module._buffers[f"U_out_bar_{adapter_name}"] = buf_out

            if owner == rank:
                del self._local_projections[name]

        self._local_projections.clear()

    # -- train lifecycle -----------------------------------------------------

    def curvature_on_train_begin(self):
        logger.info(
            "CurvatureDense: collecting static KFAC projections from %s",
            self.curvature_data_path,
        )
        self._collect_factors()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._apply_projections()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def curvature_on_train_end(self):
        self._tracker.remove_hooks()
        self._fold_projections_into_dense_weights()
        # Save the merged full model + tokenizer (equivalent to SavePeftModelCallback.on_train_end)
        peft_model_path = os.path.join(self.args.output_dir, "adapter_model")
        logger.info("CurvatureDense: saving merged model to %s", peft_model_path)
        DensePeftModelForCausalLM.save_pretrained_dense(
            self.model,
            peft_model_path,
            safe_serialization=True,
            is_main_process=self.accelerator.is_main_process,
        )
        tokenizer = self._tokenizer()
        if self.accelerator.is_main_process:
            tokenizer.save_pretrained(peft_model_path)
        # Write a "completed" marker
        if self.accelerator.is_main_process:
            with open(os.path.join(self.args.output_dir, "completed"), "a"):
                pass

    # -- post-training folding -----------------------------------------------

    def _fold_projections_into_dense_weights(self) -> None:
        """Fold projection buffers into dense_delta, removing CurvatureDense.

        After folding, each module becomes a plain Dense adapter with the
        curvature projection baked into ΔW.  This ensures that saved
        checkpoints load correctly as standard Dense adapters.
        """
        logger.info(
            "CurvatureDense: folding projection buffers into "
            "dense_delta before saving."
        )
        folded = 0
        with torch.no_grad():
            for name, module in self._target_modules.items():
                adapter_name = self._adapter_names[name]

                U_in_bar = getattr(module, f"U_in_bar_{adapter_name}")
                U_out_bar = getattr(module, f"U_out_bar_{adapter_name}")
                delta = module.dense_delta[adapter_name]

                U_in = U_in_bar.to(dtype=delta.dtype, device=delta.device)
                U_out = U_out_bar.to(dtype=delta.dtype, device=delta.device)

                if U_in.shape[1] == 0 and U_out.shape[1] == 0:
                    raise RuntimeError(
                        f"Refusing to fold empty CurvatureDense projections "
                        f"for module {name}."
                    )

                # Apply bilateral projection: (I-U_outU_out^T)·ΔW·(I-U_inU_in^T)
                projected = delta
                if U_out.shape[1] > 0:
                    projected = projected - U_out @ (U_out.T @ projected)
                if U_in.shape[1] > 0:
                    projected = projected - (projected @ U_in) @ U_in.T
                delta.copy_(projected)

                folded += 1
                module.dense_variant.pop(adapter_name, None)
                module._buffers.pop(f"U_in_bar_{adapter_name}", None)
                module._buffers.pop(f"U_out_bar_{adapter_name}", None)

        logger.info(
            "CurvatureDense: folded %d/%d Dense layers into plain "
            "adapter weights.",
            folded,
            len(self._target_modules),
        )

    # -- save (merged full model via DensePeftModelForCausalLM) --------------

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Override Trainer._save to use merged-model export.

        Since Dense ΔW has the same shape as base weights, we save a
        fully merged model (no separate adapter files).
        """
        DensePeftModelForCausalLM.save_pretrained_dense(
            self.model,
            output_dir,
            safe_serialization=True,
            is_main_process=self.accelerator.is_main_process,
            state_dict=state_dict,
        )


# =========================================================================== #
# CurvatureDenseTrainer — concrete Trainer class                              #
# =========================================================================== #

class CurvatureDenseTrainer(CurvatureDenseTrainerMixin, Trainer):
    """Trainer for CurvatureDense: full-rank adapter with KFAC projection."""
    pass


__all__ = [
    "CurvatureDense",
    "CurvatureDenseCallback",
    "CurvatureDenseTrainer",
    "CurvatureDenseTrainerMixin",
    "apply_curvature_dense",
    "get_curvature_dense_model",
]
