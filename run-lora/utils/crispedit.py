"""CrispEdit full-model baseline.

Trains the base model's projection weights in place (not a PEFT adapter). A fixed
finefineweb K-FAC cache is collected once over valid next-token CE positions
(matching the original ``layer_stats_kfac_one_pass``, *not* the Fisher/KL loss
the other curvature methods use), and every 2-D weight gradient is masked in the
Kronecker eigenbasis before the optimizer sees it.
"""

from __future__ import annotations

import logging
import os
import time
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import Trainer, TrainerCallback

from utils.curv_lora import (
    CurvatureDataCollator,
    CurvatureTextDataset,
    KFACTracker,
    get_energy_threshold,
)
from utils.projected_adamw import ProjectedAdamW, project_gradient

logger = logging.getLogger(__name__)

DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def build_projection_cache(factor_a, factor_b, energy_ratio=0.9, dtype=torch.float32):
    """``{Ua, Ub, M}`` cache for one weight, already in ``dtype`` (M stays bool).

    Stores the complement of the original ``M_keep = outer(eig(A), eig(B)) < t``:
    that kept mask is a bottom-left staircase, so its top-right high-curvature
    corner is the small one and ``project_gradient`` stays exactly equivalent
    (``grad - U_B @ ((U_B.T @ grad @ U_A) * M.T) @ U_A.T``) while the full bases
    and the almost-all-True kept mask are never materialised.
    """
    Sa, Ua = torch.linalg.eigh(factor_a.float())
    Sb, Ub = torch.linalg.eigh(factor_b.float())
    kron = torch.outer(Sa.clamp(min=0), Sb.clamp(min=0))
    drop = kron >= get_energy_threshold(kron.reshape(-1), energy_ratio)
    # Both masks are staircases: count the all-False leading rows/columns.
    first_in = int((~drop[:, -1]).sum())
    first_out = int((~drop[-1, :]).sum())
    # .contiguous(): eigh returns views and the shared all_reduce needs contiguity.
    return {
        "Ua": Ua[:, first_in:].to(dtype).contiguous(),
        "Ub": Ub[:, first_out:].to(dtype).contiguous(),
        "M": drop[first_in:, first_out:].contiguous(),
    }


def _log_cuda_memory(tag):
    if not torch.cuda.is_available():
        return
    logger.info(
        "CrispEdit: %s | allocated %.2f GiB, reserved %.2f GiB, peak %.2f GiB",
        tag,
        torch.cuda.memory_allocated() / 2**30,
        torch.cuda.memory_reserved() / 2**30,
        torch.cuda.max_memory_allocated() / 2**30,
    )


def install_gradient_projection(targets, caches):
    """Hook each target weight so its gradient is projected before accumulation.

    ``Tensor.register_hook`` is a pre-accumulate hook while DeepSpeed reduces
    from a ``register_post_accumulate_grad_hook``, so the reducer only ever sees
    the projected, full-size gradient. The caches are already on the device.
    """
    return [
        module.weight.register_hook(partial(project_gradient, cache=caches[name]))
        for name, module in targets.items()
    ]


class CrispEditTrainer(Trainer):
    def __init__(
        self,
        *args,
        target_modules=DEFAULT_TARGET_MODULES,  # same --target_modules as every other method
        crispedit_data_path=None,
        crispedit_energy_ratio=0.9,
        crispedit_collection_batch_size=1,
        crispedit_num_chunks=1,
        **kwargs,
    ):
        if not crispedit_data_path:
            raise ValueError("crispedit_data_path is required for CrispEdit.")
        model = kwargs.pop("model")

        # Freeze everything, keep the shared target projection weights trainable.
        targets = {t.strip() for t in target_modules.split(",") if t.strip()}
        for p in model.parameters():
            p.requires_grad_(False)
        self._targets = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in targets:
                module.weight.requires_grad_(True)
                self._targets[name] = module
        if not self._targets:
            raise ValueError(f"CrispEdit found no target modules in {sorted(targets)}.")

        self._er = float(crispedit_energy_ratio)
        self._data_path = crispedit_data_path
        self._coll_bs = int(crispedit_collection_batch_size)
        self._chunks = int(crispedit_num_chunks)
        self._caches = {}
        self._hooks = []
        # {weight: cache} for the non-DeepSpeed ProjectedAdamW path, filled after collection.
        self._projection_cache_map = {}

        super().__init__(*args, model=model, **kwargs)
        self.add_callback(_Callback(self))
        self._setup_tracker()

    def create_optimizer(self):
        """ProjectedAdamW without DeepSpeed; with it, the hooks above do the work."""
        if self.optimizer is None:
            if self.args.deepspeed:
                return super().create_optimizer()
            self.optimizer = ProjectedAdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                projection_cache_map=self._projection_cache_map,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
        return self.optimizer

    def _tokenizer(self):
        return getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)

    def _setup_tracker(self):
        # Round-robin split so each rank holds ~1/world_size of the A/S
        # accumulators. The chunk count comes from the *global* module count so
        # every rank runs the same number of passes: KFACTracker would otherwise
        # clamp to that rank's own count and desynchronise _share_caches.
        rank, world = self.args.process_index, self.args.world_size
        assigned = {
            name: module
            for i, (name, module) in enumerate(self._targets.items())
            if i % world == rank
        }
        self._tracker = KFACTracker(
            assigned,
            self.accelerator.device,
            max(1, min(self._chunks, len(self._targets))),
        )
        accumulator_gib = sum(
            t.numel() * t.element_size()
            for t in (*self._tracker.kfac_A.values(), *self._tracker.kfac_S.values())
        ) / 2**30
        logger.info(
            "CrispEdit: rank %d/%d collects %d/%d modules in %d chunk(s): %.1f GiB of "
            "K-FAC A/S, %.1f GiB on the GPU at a time",
            rank, world, len(assigned), len(self._targets), self._tracker.num_chunks,
            accumulator_gib, accumulator_gib / self._tracker.num_chunks,
        )
        tokenizer = self._tokenizer()
        dataset = CurvatureTextDataset(
            self._data_path, tokenizer, self.args.model_max_length
        )
        self._crispedit_loader = DataLoader(
            dataset,
            batch_size=self._coll_bs,
            sampler=SequentialSampler(dataset),
            collate_fn=CurvatureDataCollator(tokenizer),
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _ce_loss(self, inputs):
        """Plain next-token CE (labels = shifted input_ids, pad -> -100)."""
        logits = self.model(**inputs).logits[..., :-1, :].contiguous()
        labels = inputs["labels"][..., 1:].contiguous()
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
        )

    def _collect_factors(self):
        is_main = self.args.process_index == 0
        for chunk in range(self._tracker.num_chunks):
            self._tracker.activate_chunk(chunk)
            self._tracker.clear_accumulators()
            self._tracker.install_hooks()
            self.model.eval()
            try:
                for batch in tqdm(
                    self._crispedit_loader,
                    disable=not is_main,
                    desc="CrispEdit KFAC collection",
                ):
                    batch = self._prepare_inputs(batch)
                    self.model.zero_grad(set_to_none=True)
                    with self._tracker.capture(batch["attention_mask"]):
                        with self.accelerator.no_sync(self.model):
                            self.accelerator.backward(self._ce_loss(batch))
                    self.model.zero_grad(set_to_none=True)
            finally:
                self._tracker.remove_hooks()
                self.model.train()
            names = self._tracker.active_module_names()
            if is_main:
                logger.info(
                    "CrispEdit: chunk %d/%d collected, building the projection cache "
                    "for %d modules",
                    chunk + 1, self._tracker.num_chunks, len(names),
                )
            started = time.time()
            for name in names:
                n = self._tracker.kfac_n[name]
                # Normalise in place (no full-size temporary) and free the
                # accumulators as soon as this module's cache exists.
                factor_a = self._tracker.kfac_A[name].div_(n)
                factor_b = self._tracker.kfac_S[name].div_(n)
                self._caches[name] = build_projection_cache(
                    factor_a,
                    factor_b,
                    self._er,
                    # gate_proj is upcast to fp32 by train.py, so this is per module.
                    dtype=self._targets[name].weight.dtype,
                )
                self._tracker.release_module_state(name)
            if is_main:
                logger.info(
                    "CrispEdit: chunk %d/%d cache built in %.1fs",
                    chunk + 1, self._tracker.num_chunks, time.time() - started,
                )
                _log_cuda_memory(f"after chunk {chunk + 1}/{self._tracker.num_chunks}")
            self._tracker.deactivate_chunk(chunk)

    def _share_caches(self):
        if self.args.world_size == 1:
            return
        names = list(self._targets)
        device = self.accelerator.device
        rank, world = self.args.process_index, self.args.world_size

        # Basis sizes differ per module, so exchange them first. The tensors go
        # out in the module dtype: every element is non-zero on exactly one rank,
        # so this sum is exact even in bf16 (and the 0/1 mask stays exact).
        sizes = torch.zeros(len(names), 2, dtype=torch.long, device=device)
        for i, name in enumerate(names):
            if i % world == rank:
                sizes[i, 0] = self._caches[name]["Ua"].shape[1]
                sizes[i, 1] = self._caches[name]["Ub"].shape[1]
        sizes = self.accelerator.reduce(sizes, reduction="sum")

        for i, name in enumerate(names):
            module = self._targets[name]
            dtype = module.weight.dtype
            p, q = int(sizes[i, 0].item()), int(sizes[i, 1].item())
            owner = i % world == rank
            shared = {}
            for key, shape, is_mask in (
                ("Ua", (module.in_features, p), False),
                ("Ub", (module.out_features, q), False),
                ("M", (p, q), True),
            ):
                tensor = (
                    self._caches[name][key].to(device=device, dtype=dtype)
                    if owner
                    else torch.zeros(shape, dtype=dtype, device=device)
                ).contiguous()
                tensor = self.accelerator.reduce(tensor, reduction="sum")
                shared[key] = tensor > 0.5 if is_mask else tensor
            self._caches[name] = shared

    def _on_train_begin(self):
        started = time.time()
        self._collect_factors()
        if self.args.process_index == 0:
            logger.info(
                "CrispEdit: collection + cache build done in %.1fs; sharing across %d ranks",
                time.time() - started, self.args.world_size,
            )
        self._share_caches()
        self._hooks = install_gradient_projection(self._targets, self._caches)
        self._projection_cache_map.update(
            {self._targets[name].weight: cache for name, cache in self._caches.items()}
        )
        # Release calibration state (the K-FAC accumulators are already gone).
        self._tracker = None
        self._crispedit_loader = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.args.process_index == 0:
            cache_bytes = sum(
                t.numel() * t.element_size()
                for cache in self._caches.values()
                for t in cache.values()
            )
            dtype = next(iter(self._targets.values())).weight.dtype
            logger.info(
                "CrispEdit: %d target modules, %.2f GiB %s basis per rank; training starts now",
                len(self._caches), cache_bytes / 2**30, dtype,
            )
            _log_cuda_memory("after installing the projection")

    def _save_full_model(self):
        out = os.path.join(self.args.output_dir, "adapter_model")
        state_dict = None
        if self.is_deepspeed_enabled:
            # ZeRO-3 state_dict() only holds the local shard; this is the
            # collective consolidated gather (present only on a stage-3 engine).
            engine = self.model_wrapped or self.deepspeed
            gather = getattr(engine, "_zero3_consolidated_16bit_state_dict", None)
            if callable(gather):
                state_dict = gather()
        if state_dict is None:
            state_dict = self.model.state_dict()
        if self.args.should_save:
            self._save(out, state_dict={k: v.cpu() for k, v in state_dict.items()})
        if self.accelerator.is_main_process:
            self._tokenizer().save_pretrained(out)
            with open(os.path.join(self.args.output_dir, "completed"), "a"):
                pass

    def _on_train_end(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        # Drop the resident basis before saving: the consolidated ZeRO-3 gather
        # materialises a full bf16 copy of the model on the GPU, so it is worth
        # a few GiB of headroom.
        self._caches.clear()
        self._projection_cache_map.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._save_full_model()


class _Callback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer._on_train_begin()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.trainer._on_train_end()
        return control
