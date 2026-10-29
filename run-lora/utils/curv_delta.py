"""CurvDelta — full-rank adapter with KFAC bilateral curvature projection.

Provides:
- CurvDelta(DenseVariant)       — bilateral projection variant
- apply_curv_delta(model)       — attaches CurvDelta to DenseLinear layers
- get_curv_delta_model(model)   — convenience: DenseModel + CurvDelta
- CurvDeltaTrainerMixin         — KFAC collection, projection, folding
- CurvDeltaTrainer              — concrete Trainer class

Reuses KFACTracker, compute_projection_cache, CurvatureTextDataset,
CurvatureDataCollator from curv_lora.py.

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

from utils.curv_lora import (
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
# CurvDelta — DenseVariant that applies bilateral projection             #
# =========================================================================== #

class CurvDelta(DenseVariant):
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

        # ZeRO-3 gathers delta through the external-parameter hook registered
        # for the owning DenseLinear before this forward begins.
        update = F.linear(x_projected, delta)

        if U_out_bar.shape[1] > 0:
            update = update - (update @ U_out_bar) @ U_out_bar.T

        return result + update.to(result.dtype)


# =========================================================================== #
# apply_curv_delta / get_curv_delta_model                           #
# =========================================================================== #

def _get_active_adapter(module) -> Optional[str]:
    if hasattr(module, "active_adapters") and module.active_adapters:
        return module.active_adapters[0]
    if hasattr(module, "_active_adapter"):
        return module._active_adapter
    return None


def apply_curv_delta(model, adapter_name: Optional[str] = None):
    """Attach CurvDelta variant to every DenseLinear in *model*."""
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
        module.dense_variant[active] = CurvDelta()
        CurvDelta.init(module, active)
        augmented += 1

    if augmented == 0:
        raise ValueError(
            "CurvDelta found no DenseLinear modules to augment."
        )
    logger.info(
        "CurvDelta: augmented %s/%s Dense layers", augmented, total_dense
    )
    return model


def get_curv_delta_model(
    model,
    config: DenseConfig,
    adapter_name: str = "default",
    low_cpu_mem_usage: bool = False,
):
    """Construct DenseModel then attach CurvDelta variant."""
    peft_model = get_dense_model(
        model,
        config,
        adapter_name=adapter_name,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    return apply_curv_delta(peft_model, adapter_name=adapter_name)


# =========================================================================== #
# CurvDeltaCallback — hooks into Trainer lifecycle                       #
# =========================================================================== #

class CurvDeltaCallback(TrainerCallback):
    """Triggers KFAC collection at train_begin and folding at train_end.

    No ``on_optimizer_step`` hook — the trust-region trick is deprecated.
    """

    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        # ZeRO-3 is now initialised — register dense_delta params so
        # DeepSpeed auto-gathers them before DenseLinear.forward.
        self.trainer._register_zero3_external_params()
        self.trainer.curvature_on_train_begin()

    def on_train_end(self, args, state, control, **kwargs):
        self.trainer.curvature_on_train_end()


# =========================================================================== #
# CurvDeltaTrainerMixin — KFAC + projection + folding                    #
# =========================================================================== #

class CurvDeltaTrainerMixin:
    """Mixin that manages KFAC collection and projection for CurvDelta.

    Reuses the KFACTracker / compute_projection_cache machinery from
    curv_lora.py.  The Dense-specific parts are: discovering
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
                "curvature_data_path is required for CurvDelta."
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
        self.add_callback(CurvDeltaCallback(self))
        self._setup_curvature()

    # -- ZeRO helpers --------------------------------------------------------

    def _register_zero3_external_params(self) -> None:
        """Register dense_delta params for ZeRO-3 auto-gather.

        Parameters inside ``nn.ParameterDict`` are not automatically
        all-gathered by DeepSpeed ZeRO-3 because ``ParameterDict.__getitem__``
        bypasses the module-level ``__getattr__`` hook.  This method tells
        DeepSpeed to treat each ``dense_delta[adapter]`` as an external
        parameter of its owning ``DenseLinear`` module, so ZeRO-3 gathers it
        before ``DenseLinear.forward`` and reduce-scatters gradients afterward.
        """
        if not getattr(self, "is_deepspeed_enabled", False):
            return
        try:
            import deepspeed

            register_external_parameter = getattr(
                deepspeed.zero, "register_external_parameter", None
            )
            if register_external_parameter is None:
                from deepspeed.runtime.zero.partition_parameters import (
                    register_external_parameter,
                )
        except (ImportError, AttributeError):
            raise RuntimeError(
                "CurvDelta requires DeepSpeed external-parameter "
                "registration under ZeRO-3, but this DeepSpeed installation "
                "does not expose register_external_parameter."
            )

        registered = 0
        for name, module in self._target_modules.items():
            adapter_name = self._adapter_names[name]
            delta = module.dense_delta[adapter_name]
            expected_shape = (module.out_features, module.in_features)
            actual_shape = tuple(getattr(delta, "ds_shape", delta.shape))
            if actual_shape != expected_shape:
                raise RuntimeError(
                    f"CurvDelta initialized an invalid delta for {name}: "
                    f"expected {expected_shape}, got local={tuple(delta.shape)}, "
                    f"ds_shape={actual_shape}, "
                    f"ds_numel={getattr(delta, 'ds_numel', None)}."
                )
            if not delta.requires_grad:
                raise RuntimeError(
                    f"CurvDelta delta for {name} is unexpectedly frozen."
                )
            register_external_parameter(module, delta)
            registered += 1

        logger.info(
            "CurvDelta: registered %d external parameters for ZeRO-3.",
            registered,
        )

    # -- module discovery ----------------------------------------------------

    def _setup_curvature(self) -> None:
        """Discover DenseLinear modules with attached CurvDelta variants."""
        for name, module in self.model.named_modules():
            if not isinstance(module, DenseLinear):
                continue
            if not getattr(module, "dense_variant", {}):
                continue
            self._target_modules[name] = module
            self._adapter_names[name] = next(iter(module.dense_variant))

        if not self._target_modules:
            raise ValueError(
                "CurvDelta trainer found no modules with attached "
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

    # -- data loading (shared with curv_lora) ---------------------------

    def _tokenizer(self):
        tokenizer = (
            getattr(self, "processing_class", None)
            or getattr(self, "tokenizer", None)
        )
        if tokenizer is None:
            raise ValueError(
                "CurvDelta trainer requires a tokenizer/processing_class."
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
        """KL-style Fisher loss using reference model for target sampling.

        During bootstrap (pre-DeepSpeed, adapter disabled) ``self.ref_model``
        is pre-set to the same *model* so that both reference and student
        run on the base model.  During any later phase the ref_model is
        provided externally.
        """
        ref = self.ref_model if self.ref_model is not None else model
        with torch.no_grad():
            ref_outputs = ref(**inputs)
            ref_logits = ref_outputs.logits[..., :-1, :] / FISHER_TEMPERATURE
            ref_probs = F.softmax(ref_logits, dim=-1)

        outputs = model(**inputs)
        student_logits = outputs.logits[..., :-1, :]
        valid_mask = self._valid_target_mask(inputs).to(student_logits.device)
        if valid_mask.sum().item() == 0:
            raise ValueError(
                "CurvDelta Fisher collection found no valid next-token "
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

    # -- KFAC collection (adapter disabled, after DeepSpeed init) ------------

    def _collect_factors(self) -> None:
        """Collect KFAC with Dense adapters **disabled**.

        This runs from ``on_train_begin`` after DeepSpeed has installed its
        module hooks, so base Linear weights are gathered by ZeRO-3.  Dense
        adapters remain disabled for the Fisher/KFAC pass because Delta W is
        zero at initialization and is not part of the curvature estimate.
        """
        logger.info(
            "CurvDelta: collecting KFAC projections from %s "
            "(adapter disabled, post-DeepSpeed init)",
            self.curvature_data_path,
        )
        tracker = self._tracker
        self._local_projections = {}
        is_main = self.args.process_index == 0

        # Use the same (adapter-disabled) model for reference sampling.
        self.ref_model = self.model

        # Disable the exact DenseLinear modules tracked by KFAC. Wrapper-level
        # PEFT delegation is unreliable after DeepSpeed wraps the trainer model.
        for module in self._target_modules.values():
            module.enable_adapters(False)
        try:
            for chunk_idx in range(tracker.num_chunks):
                self._curvature_fisher_generator = None
                tracker.activate_chunk(chunk_idx)
                tracker.clear_accumulators()
                tracker.install_hooks()
                self.model.eval()

                chunk_names = tracker.active_module_names()
                desc = (
                    f"Bootstrapping CurvDelta KFAC "
                    f"({chunk_idx + 1}/{tracker.num_chunks})"
                    if tracker.num_chunks > 1
                    else "Bootstrapping CurvDelta KFAC"
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
                            f"CurvDelta collected no valid tokens "
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
                            f"CurvDelta produced empty projection bases "
                            f"for module {name}; KFAC factors have zero energy."
                        )
                    self._local_projections[name] = (U_in.cpu(), U_out.cpu())

                tracker.deactivate_chunk(chunk_idx)
        finally:
            for module in self._target_modules.values():
                module.enable_adapters(True)
            self.ref_model = None

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
        """Broadcast projections so every rank holds identical buffers.

        Uses ``torch.distributed.all_reduce`` because collection runs after
        DeepSpeed has initialized the process group.
        """
        device = torch.cuda.current_device()
        rank = self.args.process_index
        world_size = self.args.world_size
        module_names = list(self._target_modules.keys())

        # Phase 1: all-reduce projection shapes
        shapes = torch.zeros(len(module_names), 2, dtype=torch.long, device=device)
        for i, name in enumerate(module_names):
            if i % world_size == rank:
                U_in, U_out = self._local_projections[name]
                shapes[i, 0], shapes[i, 1] = U_in.shape[1], U_out.shape[1]
        torch.distributed.all_reduce(shapes, op=torch.distributed.ReduceOp.SUM)

        # Phase 2: all-reduce projections, write buffers
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
            torch.distributed.all_reduce(buf_in, op=torch.distributed.ReduceOp.SUM)
            module._buffers[f"U_in_bar_{adapter_name}"] = buf_in.to(ref)

            if owner == rank:
                buf_out = self._local_projections[name][1].to(
                    device=device, dtype=torch.float32
                )
            else:
                buf_out = torch.zeros(
                    module.out_features, out_cols,
                    dtype=torch.float32, device=device,
                )
            torch.distributed.all_reduce(buf_out, op=torch.distributed.ReduceOp.SUM)
            module._buffers[f"U_out_bar_{adapter_name}"] = buf_out.to(ref)

            if owner == rank:
                del self._local_projections[name]

        self._local_projections.clear()

    # -- state-dict gathering (ZeRO-3 safe) ----------------------------------

    def _gather_full_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return the full, unsharded model state dict (ZeRO-3-safe).

        Under DeepSpeed ZeRO-3 every parameter is partitioned across ranks, so
        ``self.model.state_dict()`` yields shards. Saving those produces a
        broken checkpoint (observed: 4.5MB for a 7B model that should be
        ~13GB) and downstream loaders (e.g. vLLM) fail with shape asserts.
        This mirrors HF Trainer's own ZeRO-3 save path: consolidate the 16-bit
        weights across ranks before exporting.
        """
        if not self._is_zero3(self):
            return self.model.state_dict()

        # 1) Prefer the DeepSpeed engine's consolidated 16-bit gather (the one
        #    HF Trainer uses when saving under ZeRO-3). All ranks must call it.
        engine = getattr(self, "model_wrapped", None) or getattr(self, "deepspeed", None)
        gather = getattr(engine, "_zero3_consolidated_16bit_state_dict", None)
        if callable(gather):
            try:
                gathered = gather()
                if gathered:
                    return gathered
            except Exception:
                logger.exception(
                    "CurvDelta: ZeRO-3 consolidated state dict failed, "
                    "falling back to manual per-parameter gather."
                )

        # 2) Fallback: gather each parameter explicitly. This is a collective
        #    across ranks; the full tensor is available on every rank inside
        #    the context, and only the main process uses the result.
        import deepspeed
        state_dict: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
                state_dict[name] = param.data.detach().to("cpu")
        return state_dict

    # -- train lifecycle -----------------------------------------------------

    def curvature_on_train_begin(self):
        self._collect_factors()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._apply_projections()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def curvature_on_train_end(self):
        self._tracker.remove_hooks()
        self._fold_projections_into_dense_weights()
        # Save the merged full model + tokenizer (equivalent to SavePeftModelCallback.on_train_end).
        # Under ZeRO-3 model.state_dict() is sharded, so gather full weights first.
        peft_model_path = os.path.join(self.args.output_dir, "adapter_model")
        logger.info("CurvDelta: saving merged model to %s", peft_model_path)
        DensePeftModelForCausalLM.save_pretrained_dense(
            self.model,
            peft_model_path,
            safe_serialization=True,
            is_main_process=self.accelerator.is_main_process,
            state_dict=self._gather_full_state_dict(),
        )
        tokenizer = self._tokenizer()
        if self.accelerator.is_main_process:
            tokenizer.save_pretrained(peft_model_path)
        # Write a "completed" marker
        if self.accelerator.is_main_process:
            with open(os.path.join(self.args.output_dir, "completed"), "a"):
                pass

    # -- post-training folding -----------------------------------------------

    @staticmethod
    def _is_zero3(trainer) -> bool:
        """Check whether the trainer is running under DeepSpeed ZeRO-3.

        Under the HF Trainer, ``deepspeed_plugin.deepspeed_config`` is an
        ``HfTrainerDeepSpeedConfig`` object, NOT a plain dict.  The previous
        ``isinstance(config, dict)`` check therefore silently returned False
        and the model was saved from sharded ``state_dict()`` (a 7B model
        became ~KB, breaking vLLM loading).  Use the plugin's ``zero_stage``
        property instead.
        """
        if not getattr(trainer, "is_deepspeed_enabled", False):
            return False
        try:
            plugin = getattr(getattr(trainer, "accelerator", None), "state", None)
            plugin = getattr(plugin, "deepspeed_plugin", None)
            if plugin is not None and getattr(plugin, "zero_stage", None) is not None:
                return int(plugin.zero_stage) == 3
        except Exception:
            pass
        return False

    def _fold_projections_into_dense_weights(self) -> None:
        """Fold projection buffers into dense_delta, removing CurvDelta.

        After folding, each module becomes a plain Dense adapter with the
        curvature projection baked into ΔW.  This ensures that saved
        checkpoints load correctly as standard Dense adapters.

        ZeRO-3 safety: the full bilateral projection and in-place ``copy_``
        are performed inside ``GatheredParameters`` so the parameter is
        fully assembled for the duration of the operation.  Returning a
        reference to ``param.data`` from inside the context manager (as the
        previous ``_gather_param_if_zero3`` did) produces a dangling alias
        because the parameter is re-sharded on context exit.
        """
        logger.info(
            "CurvDelta: folding projection buffers into "
            "dense_delta before saving."
        )
        folded = 0
        is_zero3 = self._is_zero3(self)
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
                        f"Refusing to fold empty CurvDelta projections "
                        f"for module {name}."
                    )

                # Apply bilateral projection: (I-U_outU_out^T)·ΔW·(I-U_inU_in^T)
                # Under ZeRO-3, delta is sharded — gather, project, and copy_
                # all inside GatheredParameters so the reference stays valid.
                if is_zero3:
                    import deepspeed
                    with deepspeed.zero.GatheredParameters([delta], modifier_rank=0):
                        full_delta = delta.data
                        projected = full_delta
                        if U_out.shape[1] > 0:
                            projected = projected - U_out @ (U_out.T @ projected)
                        if U_in.shape[1] > 0:
                            projected = projected - (projected @ U_in) @ U_in.T
                        full_delta.copy_(projected)
                else:
                    full_delta = delta.data
                    projected = full_delta
                    if U_out.shape[1] > 0:
                        projected = projected - U_out @ (U_out.T @ projected)
                    if U_in.shape[1] > 0:
                        projected = projected - (projected @ U_in) @ U_in.T
                    full_delta.copy_(projected)

                folded += 1
                module.dense_variant.pop(adapter_name, None)
                module._buffers.pop(f"U_in_bar_{adapter_name}", None)
                module._buffers.pop(f"U_out_bar_{adapter_name}", None)

        logger.info(
            "CurvDelta: folded %d/%d Dense layers into plain "
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
            state_dict=(
                state_dict
                if state_dict is not None
                else self._gather_full_state_dict()
            ),
        )


# =========================================================================== #
# CurvDeltaTrainer — concrete Trainer class                              #
# =========================================================================== #

class CurvDeltaTrainer(CurvDeltaTrainerMixin, Trainer):
    """Trainer for CurvDelta: full-rank adapter with KFAC projection."""
    pass


__all__ = [
    "CurvDelta",
    "CurvDeltaCallback",
    "CurvDeltaTrainer",
    "CurvDeltaTrainerMixin",
    "apply_curv_delta",
    "get_curv_delta_model",
]
