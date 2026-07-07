import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm
from transformers import Trainer, TrainerCallback

from peft.tuners.lora.layer import LoraLayer, LoraVariant

logger = logging.getLogger(__name__)

FISHER_SAMPLE_SEED = 0
FISHER_TEMPERATURE = 1.0


class CurvatureLora(LoraVariant):
    """LoRA variant that projects LoRA updates away from high-curvature KFAC bases."""

    @staticmethod
    def init(module: LoraLayer, adapter_name: str, **kwargs) -> None:
        base_weight = module.base_layer.weight if hasattr(module, "base_layer") else module.weight
        module.register_buffer(
            f"U_in_bar_{adapter_name}",
            base_weight.new_zeros((module.in_features, 0), dtype=base_weight.dtype),
        )
        module.register_buffer(
            f"U_out_bar_{adapter_name}",
            base_weight.new_zeros((module.out_features, 0), dtype=base_weight.dtype),
        )

    @staticmethod
    def _projection_buffers(module: LoraLayer, active_adapter: str):
        U_in_bar = getattr(module, f"U_in_bar_{active_adapter}", None)
        U_out_bar = getattr(module, f"U_out_bar_{active_adapter}", None)
        if U_in_bar is None or U_out_bar is None:
            raise RuntimeError(
                f"CurvatureLoRA adapter {active_adapter!r} is active, but projection buffers are missing."
            )
        return U_in_bar, U_out_bar

    @staticmethod
    def forward(
        module: LoraLayer,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        U_in_bar, U_out_bar = CurvatureLora._projection_buffers(module, active_adapter)
        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]
        weight_A = lora_A.weight
        weight_B = lora_B.weight

        compute_dtype = weight_A.dtype
        U_in_bar = U_in_bar.to(dtype=compute_dtype)
        U_out_bar = U_out_bar.to(dtype=compute_dtype)

        A_eff = weight_A - (weight_A @ U_in_bar) @ U_in_bar.T
        B_eff = weight_B - U_out_bar @ (U_out_bar.T @ weight_B)

        x_in = dropout(x)
        if x_in.dtype != compute_dtype:
            x_in = x_in.to(compute_dtype)
        h = F.linear(F.linear(x_in, A_eff), B_eff)
        return result + (h * scaling).to(result.dtype)

    @staticmethod
    def _compute_delta_weight(module: LoraLayer, active_adapter: str) -> torch.Tensor:
        U_in_bar, U_out_bar = CurvatureLora._projection_buffers(module, active_adapter)
        weight_A = module.lora_A[active_adapter].weight
        weight_B = module.lora_B[active_adapter].weight
        scaling = module.scaling[active_adapter]
        dtype = weight_B.dtype

        cast_to_fp32 = weight_B.device.type == "cpu" and dtype in (torch.float16, torch.bfloat16)
        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()
            U_in_bar = U_in_bar.float()
            U_out_bar = U_out_bar.float()
        else:
            U_in_bar = U_in_bar.to(dtype=dtype)
            U_out_bar = U_out_bar.to(dtype=dtype)

        delta = weight_B @ weight_A
        if U_out_bar.shape[1] > 0:
            delta = delta - U_out_bar @ (U_out_bar.T @ delta)
        if U_in_bar.shape[1] > 0:
            delta = delta - (delta @ U_in_bar) @ U_in_bar.T
        delta = delta * scaling
        return delta.to(dtype=dtype) if cast_to_fp32 else delta

    @staticmethod
    def merge_safe(module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        delta = CurvatureLora._compute_delta_weight(module, active_adapter)
        return orig_weight + delta.to(orig_weight.dtype)

    @staticmethod
    def merge_unsafe(module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor) -> None:
        delta = CurvatureLora._compute_delta_weight(module, active_adapter)
        orig_weight.data += delta.to(orig_weight.dtype)

    @staticmethod
    def unmerge(module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        delta = CurvatureLora._compute_delta_weight(module, active_adapter)
        return orig_weight - delta.to(orig_weight.dtype)


def _get_active_adapter(module) -> Optional[str]:
    if hasattr(module, "active_adapters") and module.active_adapters:
        return module.active_adapters[0]
    if hasattr(module, "_active_adapter"):
        return module._active_adapter
    return None


def apply_curvature_lora(model, adapter_name: Optional[str] = None):
    from peft.tuners.lora.layer import Linear as PeftLoraLinear

    augmented = 0
    total_lora = 0
    for _, module in model.named_modules():
        if not isinstance(module, PeftLoraLinear):
            continue
        total_lora += 1
        active = adapter_name or _get_active_adapter(module)
        if active is None:
            raise ValueError("CurvatureLoRA could not determine the active PEFT adapter name.")
        if active in getattr(module, "lora_variant", {}):
            raise ValueError(f"Adapter {active!r} already has a LoRA variant attached; CurvatureLoRA expects plain LoRA.")
        module.lora_variant[active] = CurvatureLora()
        CurvatureLora.init(module, active)
        augmented += 1

    if augmented == 0:
        raise ValueError("CurvatureLoRA found no PEFT LoRA Linear modules to augment.")
    logger.info("CurvatureLoRA: augmented %s/%s LoRA layers", augmented, total_lora)
    return model


def get_energy_threshold(eigenvalues: torch.Tensor, energy_ratio: float) -> float:
    total_energy = eigenvalues.sum()
    if total_energy == 0:
        return float("inf")

    sorted_eigvals, _ = torch.sort(eigenvalues, descending=True)
    cumulative_energy = torch.cumsum(sorted_eigvals, dim=0)
    energy_ratios = cumulative_energy / total_energy
    rank = torch.searchsorted(energy_ratios, energy_ratio).item() + 1
    rank = min(rank, len(sorted_eigvals))
    return sorted_eigvals[rank - 1].item()


def compute_projection_cache(factor: torch.Tensor, energy_ratio: float = 0.9) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(factor.float())
    eigenvalues = eigenvalues.clamp(min=0)

    threshold = get_energy_threshold(eigenvalues, energy_ratio)
    if threshold == float("inf"):
        return torch.zeros(factor.shape[0], 0, dtype=torch.float32, device=factor.device)
    return eigenvectors[:, eigenvalues >= threshold]


class CurvatureTextDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int):
        self.examples = []
        jsonl_path = Path(path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Curvature anchor set not found: {jsonl_path}")
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in curvature anchor record {line_no}: {exc}") from exc
                text = record.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Curvature anchor record {line_no} must contain a non-empty string field 'text'.")
                encoded = tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                    add_special_tokens=True,
                )
                if encoded["input_ids"]:
                    self.examples.append(encoded)
        if not self.examples:
            raise ValueError(f"Curvature anchor set is empty after filtering: {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class CurvatureDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        batch = self.tokenizer.pad(instances, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


class _DisabledAdapterModel:
    def __init__(self, model):
        object.__setattr__(self, "_model", model)

    def __call__(self, *args, **kwargs):
        was_training = self._model.training
        self._model.eval()
        try:
            with self._model.disable_adapter():
                return self._model(*args, **kwargs)
        finally:
            if was_training:
                self._model.train()

    def __getattr__(self, name):
        return getattr(self._model, name)


class KFACTracker:
    def __init__(self, target_modules: Dict[str, torch.nn.Module], device: torch.device, num_chunks: int = 1):
        if not target_modules:
            raise ValueError("KFACTracker requires at least one target module.")
        if num_chunks < 1:
            raise ValueError(f"num_chunks must be >= 1, got {num_chunks}.")

        self.target_modules = target_modules
        self._compute_device = device
        self._storage_device = self._compute_device if num_chunks <= 1 else torch.device("cpu")
        self._handles = []
        self._capturing = False
        self._current_mask: Optional[torch.Tensor] = None
        self._saved_acts: Dict[str, list] = {n: [] for n in target_modules}
        self.kfac_A: Dict[str, torch.Tensor] = {}
        self.kfac_S: Dict[str, torch.Tensor] = {}
        self.kfac_n: Dict[str, int] = {}

        for name, module in target_modules.items():
            self.kfac_A[name] = torch.zeros(module.in_features, module.in_features, dtype=torch.float32, device=self._storage_device)
            self.kfac_S[name] = torch.zeros(module.out_features, module.out_features, dtype=torch.float32, device=self._storage_device)
            self.kfac_n[name] = 0

        module_names = list(target_modules.keys())
        num_chunks = min(num_chunks, len(module_names))
        chunk_size = (len(module_names) + num_chunks - 1) // num_chunks
        self._chunks: List[List[str]] = [module_names[i : i + chunk_size] for i in range(0, len(module_names), chunk_size)]
        self._current_chunk: Optional[int] = None

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def activate_chunk(self, chunk_idx: int) -> None:
        self._current_chunk = chunk_idx
        self._move_chunk_state(chunk_idx, self._compute_device)

    def deactivate_chunk(self, chunk_idx: int) -> None:
        self.remove_hooks()
        self._move_chunk_state(chunk_idx, self._storage_device)
        self._current_chunk = None

    def _move_chunk_state(self, chunk_idx: int, device: torch.device) -> None:
        for name in self._chunks[chunk_idx]:
            self.kfac_A[name] = self.kfac_A[name].to(device)
            self.kfac_S[name] = self.kfac_S[name].to(device)

    def active_module_names(self) -> list:
        if self._current_chunk is None:
            return list(self.target_modules.keys())
        return self._chunks[self._current_chunk]

    def install_hooks(self) -> None:
        for name in self.active_module_names():
            module = self.target_modules[name]
            self._handles.append(module.register_forward_pre_hook(self._make_fwd_hook(name)))
            self._handles.append(module.register_full_backward_pre_hook(self._make_bwd_hook(name)))

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._clear_transient_state()

    @contextmanager
    def capture(self, mask: torch.Tensor):
        self._current_mask = mask
        self._capturing = True
        try:
            yield
        finally:
            self._capturing = False
            self._current_mask = None
            self._clear_transient_state()

    def clear_accumulators(self) -> None:
        for name in self.active_module_names():
            self.kfac_A[name].zero_()
            self.kfac_S[name].zero_()
            self.kfac_n[name] = 0

    def _make_fwd_hook(self, name: str):
        def hook(module, args):
            if not self._capturing or not torch.is_grad_enabled():
                return
            self._saved_acts[name].append(args[0].detach())

        return hook

    def _make_bwd_hook(self, name: str):
        def hook(module, grad_output):
            if not self._capturing:
                return
            if self._current_mask is None:
                raise RuntimeError("KFAC backward hook ran without an active attention mask.")
            if not self._saved_acts[name]:
                raise RuntimeError(f"KFAC backward hook for {name!r} ran without a captured forward activation.")
            if not grad_output or grad_output[0] is None:
                raise RuntimeError(f"KFAC backward hook for {name!r} received no output gradient.")
            acts = self._saved_acts[name].pop()
            grads = grad_output[0].detach()
            valid_mask = self._current_mask[:, :-1].bool()
            acts_f = acts[:, :-1, :][valid_mask].to(dtype=torch.float32)
            grads_f = grads[:, :-1, :][valid_mask].to(dtype=torch.float32)
            self.kfac_A[name].addmm_(acts_f.T, acts_f)
            self.kfac_S[name].addmm_(grads_f.T, grads_f)
            self.kfac_n[name] += acts_f.shape[0]

        return hook

    def _clear_transient_state(self) -> None:
        for acts in self._saved_acts.values():
            acts.clear()


class CurvatureCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer.curvature_on_train_begin()

    def on_train_end(self, args, state, control, **kwargs):
        self.trainer.curvature_on_train_end()


class CurvatureTrainerMixin:
    def __init__(self, *args, curvature_data_path: str, curvature_energy_ratio: float = 0.9, curvature_collection_batch_size: int = 1, curvature_num_chunks: int = 1, **kwargs):
        if not curvature_data_path:
            raise ValueError("curvature_data_path is required for CurvatureLoRA.")
        if not 0 < float(curvature_energy_ratio) <= 1:
            raise ValueError(f"curvature_energy_ratio must be in (0, 1], got {curvature_energy_ratio}.")
        if int(curvature_collection_batch_size) < 1:
            raise ValueError(f"curvature_collection_batch_size must be >= 1, got {curvature_collection_batch_size}.")
        if int(curvature_num_chunks) < 1:
            raise ValueError(f"curvature_num_chunks must be >= 1, got {curvature_num_chunks}.")

        self.curvature_data_path = curvature_data_path
        self.curvature_energy_ratio = float(curvature_energy_ratio)
        self.curvature_collection_batch_size = int(curvature_collection_batch_size)
        self.curvature_num_chunks = int(curvature_num_chunks)
        self._target_modules: Dict[str, torch.nn.Module] = {}
        self._adapter_names: Dict[str, str] = {}
        self._local_projections: Dict[str, tuple] = {}
        self._curvature_fisher_generator: Optional[torch.Generator] = None
        self.ref_model = None
        super().__init__(*args, **kwargs)
        self._fail_on_zero3()
        self.add_callback(CurvatureCallback(self))
        self._setup_curvature()

    def _fail_on_zero3(self) -> None:
        if not getattr(self, "is_deepspeed_enabled", False):
            return
        plugin = getattr(getattr(self, "accelerator", None), "state", None)
        plugin = getattr(plugin, "deepspeed_plugin", None)
        config = getattr(plugin, "deepspeed_config", None)
        if not isinstance(config, dict):
            raise RuntimeError("CurvatureLoRA could not inspect DeepSpeed config; refusing to run without confirming ZeRO stage.")
        stage = config.get("zero_optimization", {}).get("stage")
        if str(stage) == "3":
            raise ValueError("CurvatureLoRA supports DeepSpeed ZeRO-2 only; ZeRO-3 is intentionally unsupported.")

    def _setup_curvature(self) -> None:
        for name, module in self.model.named_modules():
            if not getattr(module, "lora_variant", {}):
                continue
            self._target_modules[name] = module
            self._adapter_names[name] = next(iter(module.lora_variant))
        if not self._target_modules:
            raise ValueError("CurvatureLoRA trainer found no modules with attached curvature variants.")

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

    def _tokenizer(self):
        tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("CurvatureLoRA trainer requires a tokenizer/processing_class.")
        return tokenizer

    def _build_curvature_loader(self) -> DataLoader:
        tokenizer = self._tokenizer()
        dataset = CurvatureTextDataset(self.curvature_data_path, tokenizer, self.args.model_max_length)
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

    def _get_curvature_fisher_generator(self, device: torch.device) -> torch.Generator:
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
        if self.ref_model is None:
            self.ref_model = _DisabledAdapterModel(model)
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)
            ref_logits = ref_outputs.logits[..., :-1, :] / FISHER_TEMPERATURE
            ref_probs = F.softmax(ref_logits, dim=-1)

        outputs = model(**inputs)
        student_logits = outputs.logits[..., :-1, :]
        valid_mask = self._valid_target_mask(inputs).to(student_logits.device)
        if valid_mask.sum().item() == 0:
            raise ValueError("CurvatureLoRA Fisher collection found no valid next-token targets in a batch.")
        flat_ref_probs = ref_probs.to(student_logits.device)[valid_mask]
        generator = self._get_curvature_fisher_generator(student_logits.device)
        sampled_targets = torch.multinomial(
            flat_ref_probs.float(),
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(-1)
        flat_student_logits = student_logits[valid_mask]
        return F.cross_entropy(flat_student_logits, sampled_targets, reduction="mean")

    def curvature_on_train_begin(self):
        logger.info("CurvatureLoRA: collecting static KFAC/Fisher projections from %s", self.curvature_data_path)
        self._collect_factors()
        self._apply_projections()

    def curvature_on_train_end(self):
        self._tracker.remove_hooks()
        self._fold_projections_into_lora_weights()

    def _collect_factors(self) -> None:
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
                f"Collecting CurvatureLoRA KFAC factors ({chunk_idx + 1}/{tracker.num_chunks})"
                if tracker.num_chunks > 1
                else "Collecting CurvatureLoRA KFAC factors"
            )
            try:
                pbar = tqdm(self._curvature_loader, desc=desc, disable=not is_main, dynamic_ncols=True)
                for batch in pbar:
                    batch = self._prepare_inputs(batch)
                    self.model.zero_grad(set_to_none=True)
                    with tracker.capture(batch["attention_mask"]):
                        loss = self._compute_curvature_fisher_loss(self.model, batch)
                        with self.accelerator.no_sync(self.model):
                            self.accelerator.backward(loss)
                    self.model.zero_grad(set_to_none=True)
                    pbar.set_postfix({"fisher_loss": f"{loss.item():.4f}"})
            finally:
                tracker.remove_hooks()
                self.model.train()

            for name in chunk_names:
                if tracker.kfac_n[name] == 0:
                    raise ValueError(f"CurvatureLoRA collected no valid tokens for module {name}.")
                tracker.kfac_A[name] /= tracker.kfac_n[name]
                tracker.kfac_S[name] /= tracker.kfac_n[name]
                U_in = compute_projection_cache(tracker.kfac_A[name], energy_ratio=self.curvature_energy_ratio)
                U_out = compute_projection_cache(tracker.kfac_S[name], energy_ratio=self.curvature_energy_ratio)
                if U_in.shape[1] == 0 and U_out.shape[1] == 0:
                    raise ValueError(
                        f"CurvatureLoRA produced empty projection bases for module {name}; "
                        "KFAC factors have zero energy, so this run would silently become plain LoRA."
                    )
                self._local_projections[name] = (U_in.cpu(), U_out.cpu())
            tracker.deactivate_chunk(chunk_idx)

    def _apply_projections(self) -> None:
        projections = self._distributed_share_projections() if self.args.world_size > 1 else self._local_projections
        for name, module in self._target_modules.items():
            adapter_name = self._adapter_names[name]
            U_in_bar, U_out_bar = projections[name]
            ref = module.lora_A[adapter_name].weight
            module._buffers[f"U_in_bar_{adapter_name}"] = U_in_bar.to(ref)
            module._buffers[f"U_out_bar_{adapter_name}"] = U_out_bar.to(ref)

    def _distributed_share_projections(self) -> Dict[str, tuple]:
        device = self.accelerator.device
        rank = self.args.process_index
        world_size = self.args.world_size
        module_names = list(self._target_modules.keys())
        shapes = torch.zeros(len(module_names), 2, dtype=torch.long, device=device)
        for i, name in enumerate(module_names):
            if i % world_size == rank:
                U_in, U_out = self._local_projections[name]
                shapes[i, 0] = U_in.shape[1]
                shapes[i, 1] = U_out.shape[1]
        shapes = self.accelerator.reduce(shapes, reduction="sum")

        result: Dict[str, tuple] = {}
        for i, name in enumerate(module_names):
            owner = i % world_size
            module = self._target_modules[name]
            in_cols = int(shapes[i, 0].item())
            out_cols = int(shapes[i, 1].item())
            if owner == rank:
                buf_in = self._local_projections[name][0].to(device=device, dtype=torch.float32)
            else:
                buf_in = torch.zeros(module.in_features, in_cols, dtype=torch.float32, device=device)
            buf_in = self.accelerator.reduce(buf_in, reduction="sum")
            if owner == rank:
                buf_out = self._local_projections[name][1].to(device=device, dtype=torch.float32)
            else:
                buf_out = torch.zeros(module.out_features, out_cols, dtype=torch.float32, device=device)
            buf_out = self.accelerator.reduce(buf_out, reduction="sum")
            result[name] = (buf_in, buf_out)
        return result

    def _fold_projections_into_lora_weights(self) -> None:
        logger.info("CurvatureLoRA: folding projection buffers into LoRA weights before saving.")
        folded = 0
        with torch.no_grad():
            for name, module in self._target_modules.items():
                adapter_name = self._adapter_names[name]
                U_in_bar = getattr(module, f"U_in_bar_{adapter_name}")
                U_out_bar = getattr(module, f"U_out_bar_{adapter_name}")
                weight_A = module.lora_A[adapter_name].weight
                weight_B = module.lora_B[adapter_name].weight
                U_in = U_in_bar.to(dtype=weight_A.dtype, device=weight_A.device)
                U_out = U_out_bar.to(dtype=weight_B.dtype, device=weight_B.device)
                if U_in.shape[1] == 0 and U_out.shape[1] == 0:
                    raise RuntimeError(f"Refusing to fold empty CurvatureLoRA projections for module {name}.")
                if U_in.shape[1] > 0:
                    weight_A.copy_(weight_A - (weight_A @ U_in) @ U_in.T)
                if U_out.shape[1] > 0:
                    weight_B.copy_(weight_B - U_out @ (U_out.T @ weight_B))
                folded += 1
                module.lora_variant.pop(adapter_name, None)
                module._buffers.pop(f"U_in_bar_{adapter_name}", None)
                module._buffers.pop(f"U_out_bar_{adapter_name}", None)
        logger.info("CurvatureLoRA: folded %d/%d LoRA layers into plain adapter weights.", folded, len(self._target_modules))


class CurvatureTrainer(CurvatureTrainerMixin, Trainer):
    pass
