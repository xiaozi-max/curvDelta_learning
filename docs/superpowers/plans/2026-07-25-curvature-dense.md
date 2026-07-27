# CurvatureDense Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CurvatureDense — the full-rank counterpart of CurvatureLoRA — to the lr-matters-lora experiment pipeline. CurvatureDense replaces the low-rank LoRA adapter (ΔW = B·A) with a full-rank additive adapter (ΔW ∈ R^{d_out×d_in}) while applying the same KFAC-based bilateral curvature projection to constrain updates away from high-curvature directions.

**Tech Stack:** Python 3.12, Hugging Face Transformers 4.56.0, PEFT 0.17.1, DeepSpeed ZeRO-2, Bash experiment scripts.

---

## Global Constraints

- Use Python 3.12 and existing dataclass-based argument definitions in `run-lora/train.py`.
- Compatible with **peft==0.17.1** — key version differences from the reference implementation (peft==0.19.1) are called out explicitly in each task.
- Static KFAC collection only (no refresh/EMA). The trust-region trick (`curvature_alpha_tr` / `curvature_delta_tr`) is **deprecated and will NOT be implemented** for CurvatureDense.
- Keep large generated datasets, checkpoints, adapters, and experiment outputs out of git.
- Do not commit Hugging Face tokens, W&B keys, local storage paths, downloaded datasets, or model weights.
- Do not create git commits unless the user explicitly requests them.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                  lr-matters-lora/run-lora/                    │
├──────────────────────────────────────────────────────────────┤
│  NEW: utils/dense_peft.py          Dense PEFT infrastructure │
│  NEW: utils/curvature_dense.py     CurvatureDense variant    │
│  MODIFY: utils/curvature_lora.py   Extend CurvatureTrainer   │
│  MODIFY: train.py                  Add Dense dispatch        │
│  MODIFY: scripts/train.sh          Add CurvatureDense branch │
│  MODIFY: scripts/test.sh           Add CurvatureDense branch │
│  MODIFY: scripts/*/math_curvature_dense.sh  Experiment scripts│
└──────────────────────────────────────────────────────────────┘
```

### Class Hierarchy

```
peft.tuners.tuners_utils.BaseTunerLayer
    └── DenseLayer                          ← NEW (in dense_peft.py)
            ├── dense_delta: ParameterDict  ← trainable full-rank ΔW
            ├── dense_variant: dict         ← {adapter_name: DenseVariant}
            └── DenseLinear(nn.Module, DenseLayer)  ← replaces nn.Linear

DenseVariant (abstract)                     ← NEW (in dense_peft.py)
    └── CurvatureDense                      ← NEW (in curvature_dense.py)

peft.tuners.tuners_utils.BaseTuner
    └── DenseModel                          ← NEW (in dense_peft.py)

CurvatureTrainerMixin                       ← EXISTING (in curvature_lora.py)
    └── CurvatureDenseTrainerMixin          ← NEW (in curvature_dense.py)

CurvatureTrainer(CurvatureTrainerMixin, Trainer)  ← EXISTING
CurvatureDenseTrainer(CurvatureDenseTrainerMixin, Trainer)  ← NEW
```

---

## peft 0.17.1 vs 0.19.1: Critical Version Differences

Before implementing, understand these incompatibilities between the reference implementation (peft 0.19.1) and the target (peft 0.17.1):

### 1. `BaseTuner._create_and_replace` Signature

| peft 0.19.1 | peft 0.17.1 |
|---|---|
| `(self, config, adapter_name, target, target_name, parent, current_key, **optional_kwargs)` | Same signature. `current_key` exists in both. However, the **abstract base method** in 0.17.1 may also include an optional `parameter_name` parameter. |

**Mitigation**: Use keyword argument for `current_key` and add `**optional_kwargs` to absorb any extra params.

### 2. `BaseTunerLayer.set_adapter` with `inference_mode`

| peft 0.19.1 | peft 0.17.1 |
|---|---|
| `set_adapter(adapter_names, inference_mode=False)` exists | `set_adapter` exists but may NOT have `inference_mode` parameter |

**Mitigation**: In `DenseLayer.update_layer()`, use a try/except or `getattr` pattern:
```python
try:
    self.set_adapter(self.active_adapters, inference_mode=inference_mode)
except TypeError:
    self.set_adapter(self.active_adapters)
```

### 3. `PeftConfig.inference_mode` Field

| peft 0.19.1 | peft 0.17.1 |
|---|---|
| `inference_mode` is a field in `PeftConfig` | `inference_mode` may NOT be a field; it could be a property or absent |

**Mitigation**: In `DenseModel._create_and_replace()`, use `getattr(dense_config, 'inference_mode', False)` instead of `dense_config.inference_mode`.

### 4. `DenseModel._replace_module` Signature

| peft 0.19.1 | peft 0.17.1 |
|---|---|
| `(parent, target_name, new_module, child)` — 4 args | Same — 4 args in both |

No mitigation needed.

### 5. `BaseTuner.active_adapter` Property

| peft 0.19.1 | peft 0.17.1 |
|---|---|
| Available as both `active_adapter` (singular) and `active_adapters` (list) | Same |

No mitigation needed. However, in `_create_and_replace`, 0.17.1 may use `active_adapters` (list) rather than `active_adapter` (str). Use:
```python
if adapter_name not in self.active_adapters:
    new_module.requires_grad_(False)
```

### 6. `BaseTunerLayer.active_adapter` vs `_active_adapter`

| peft 0.19.1 | peft 0.17.1 |
|---|---|
| `active_adapter` property returns first from `active_adapters` list | Same |

No mitigation needed.

### 7. `BaseTuner.inject_adapter` Method

In both versions, `inject_adapter` calls `_create_and_replace` with `current_key=key`. This is the standard PEFT injection path. Our `DenseModel` uses this path indirectly via `DensePeftModelForCausalLM.__init__` → `BaseTuner.__init__` → `inject_adapter`.

---

## Task 1: Create Dense PEFT Infrastructure

**Files:**
- Create: `run-lora/utils/dense_peft.py`

**Interfaces:**
- Produces: `DenseVariant` — abstract base for Dense adapter variants
- Produces: `DenseLayer(BaseTunerLayer)` — full-rank adapter layer
- Produces: `DenseLinear(nn.Module, DenseLayer)` — concrete module replacing `nn.Linear`
- Produces: `DenseConfig(PeftConfig)` — configuration dataclass (`peft_type="DENSE"`)
- Produces: `DenseModel(BaseTuner)` — PEFT tuner that replaces Linear with DenseLinear
- Produces: `DensePeftModelForCausalLM(PeftModelForCausalLM)` — save wrapper
- Produces: `register_dense_peft_method()` — registers DENSE into PEFT dispatch maps
- Produces: `get_dense_model(model, config)` — convenience constructor

### Step 1: Implement `DenseVariant` (abstract base)

```python
class DenseVariant:
    """Extension point for variants that transform a dense adapter update."""

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
```

### Step 2: Implement `DenseLayer(BaseTunerLayer)`

Key methods:

```python
class DenseLayer(BaseTunerLayer):
    adapter_layer_names = ("dense_delta",)
    other_param_names = ()

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.dense_delta = nn.ParameterDict({})
        self.dense_variant: dict[str, DenseVariant] = {}
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if not isinstance(base_layer, nn.Linear):
            raise ValueError(
                f"Unsupported layer type {type(base_layer)}. Dense supports torch.nn.Linear only."
            )
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        base_layer.requires_grad_(False)

    def update_layer(self, adapter_name, inference_mode=False, **kwargs):
        base_weight = self.get_base_layer().weight
        self.dense_delta[adapter_name] = nn.Parameter(
            base_weight.new_zeros(base_weight.shape), requires_grad=True
        )
        self._move_adapter_to_device_of_base_layer(adapter_name)
        # ★ peft 0.17.1 COMPAT: set_adapter may not accept inference_mode
        try:
            self.set_adapter(self.active_adapters, inference_mode=inference_mode)
        except TypeError:
            self.set_adapter(self.active_adapters)

    def get_delta_weight(self, adapter_name, delta=None):
        delta = self.dense_delta[adapter_name] if delta is None else delta
        variant = self.dense_variant.get(adapter_name)
        return variant.get_delta_weight(self, adapter_name, delta) if variant else delta

    def merge(self, safe_merge=False, adapter_names=None):
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
                    raise ValueError(f"NaNs or Infs detected merging Dense adapter {active_adapter!r}.")
                base_layer.weight.data = orig
            else:
                base_layer.weight.data += delta.to(base_layer.weight.dtype)
            self.merged_adapters.append(active_adapter)

    def unmerge(self):
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while self.merged_adapters:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self.dense_delta:
                continue
            weight = self.get_base_layer().weight
            weight.data -= self.get_delta_weight(active_adapter).to(weight.dtype)

    def forward(self, x, *args, **kwargs):
        adapter_names = kwargs.pop("adapter_names", None)
        if adapter_names is not None:
            raise NotImplementedError("Dense adapters do not support per-example adapter_names.")
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
                        self, active_adapter=active_adapter, x=x, result=result, **kwargs
                    )
                else:
                    delta = self.dense_delta[active_adapter]
                    result = result + F.linear(x.to(delta.dtype), delta).to(result_dtype)
            result = result.to(result_dtype)
        return result.to(previous_dtype)
```

### Step 3: Implement `DenseLinear(nn.Module, DenseLayer)`

```python
class DenseLinear(nn.Module, DenseLayer):
    def __init__(self, base_layer, adapter_name, inference_mode=False, **kwargs):
        super().__init__()             # nn.Module.__init__
        DenseLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, inference_mode=inference_mode)

    def forward(self, x, *args, **kwargs):
        return DenseLayer.forward(self, x, *args, **kwargs)
```

### Step 4: Implement `DenseConfig(PeftConfig)`

```python
DENSE_PEFT_TYPE = "DENSE"

@dataclass
class DenseConfig(PeftConfig):
    target_modules: Optional[Union[list[str], str]] = field(default=None)
    exclude_modules: Optional[Union[list[str], str]] = field(default=None)
    layers_to_transform: Optional[Union[list[int], int]] = field(default=None)
    layers_pattern: Optional[Union[list[str], str]] = field(default=None)
    bias: str = field(default="none")
    modules_to_save: Optional[list[str]] = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = DENSE_PEFT_TYPE
        if self.bias != "none":
            raise ValueError("Dense adapters do not support training bias parameters.")
        if self.modules_to_save:
            raise ValueError("Dense adapters do not support modules_to_save.")
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list)
            else self.target_modules
        )
```

### Step 5: Implement `DenseModel(BaseTuner)`

```python
class DenseModel(BaseTuner):
    prefix = "dense_"
    tuner_layer_cls = DenseLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        dense_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,           # ← peft 0.17.1 includes current_key
        **optional_kwargs,     # ← absorb any extra (e.g. parameter_name in 0.17.1)
    ):
        if current_key is None:
            raise ValueError("Current key cannot be None.")

        if isinstance(target, DenseLayer):
            target.update_layer(
                adapter_name,
                # ★ peft 0.17.1 COMPAT: inference_mode may not be a PeftConfig field
                inference_mode=getattr(dense_config, "inference_mode", False),
            )
            return

        new_module = self._create_new_module(
            adapter_name,
            target,
            inference_mode=getattr(dense_config, "inference_mode", False),
        )
        # ★ peft 0.17.1 COMPAT: use active_adapters (list) not active_adapter
        if adapter_name not in self.active_adapters:
            new_module.requires_grad_(False)
        self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(adapter_name, target, inference_mode=False):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target
        if not isinstance(target_base_layer, torch.nn.Linear):
            raise ValueError(f"Dense supports torch.nn.Linear only.")
        if isinstance(target, BaseTunerLayer) and not isinstance(target, DenseLayer):
            raise ValueError("Dense cannot be mixed with another PEFT tuner layer.")
        return DenseLinear(target, adapter_name, inference_mode=inference_mode)
```

### Step 6: Implement `DensePeftModelForCausalLM`

```python
class DensePeftModelForCausalLM(PeftModelForCausalLM):
    """Causal-LM PEFT wrapper that saves a merged full model."""

    def _merged_base_state_dict(self, state_dict=None):
        source = self.state_dict() if state_dict is None else state_dict
        dense_modules = {
            name: module
            for name, module in self.base_model.model.named_modules()
            if isinstance(module, DenseLinear)
        }
        # Find root prefix
        module_name = next(iter(dense_modules))
        suffix = f"{module_name}.base_layer.weight"
        root = next(key[: -len(suffix)] for key in source if key.endswith(suffix))

        merged = {}
        for key, value in source.items():
            if not key.startswith(root):
                continue
            plain_key = key[len(root):]
            # Skip adapter-only keys
            if ".dense_delta." in plain_key or ".U_in_bar_" in plain_key or ".U_out_bar_" in plain_key:
                continue
            merged[plain_key.replace(".base_layer.", ".")] = value

        for name, module in dense_modules.items():
            weight = source[f"{root}{name}.base_layer.weight"].clone()
            if not module.merged:
                for adapter_name in module.active_adapters:
                    delta = source[f"{root}{name}.dense_delta.{adapter_name}"]
                    weight += module.get_delta_weight(adapter_name, delta).to(weight.dtype)
            merged[f"{name}.weight"] = weight
        return merged

    def save_pretrained(self, save_directory, safe_serialization=True,
                        is_main_process=True, state_dict=None, **kwargs):
        return self.base_model.model.save_pretrained(
            save_directory,
            state_dict=self._merged_base_state_dict(state_dict),
            safe_serialization=safe_serialization,
            is_main_process=is_main_process,
            **kwargs,
        )
```

### Step 7: Implement registration and convenience constructor

```python
def register_dense_peft_method():
    from peft.mapping import (
        PEFT_TYPE_TO_CONFIG_MAPPING,
        PEFT_TYPE_TO_PREFIX_MAPPING,
        PEFT_TYPE_TO_TUNER_MAPPING,
    )
    entries = (
        (PEFT_TYPE_TO_CONFIG_MAPPING, DENSE_PEFT_TYPE, DenseConfig),
        (PEFT_TYPE_TO_TUNER_MAPPING, DENSE_PEFT_TYPE, DenseModel),
        (PEFT_TYPE_TO_PREFIX_MAPPING, DENSE_PEFT_TYPE, DenseModel.prefix),
    )
    for mapping, key, value in entries:
        existing = mapping.get(key)
        if existing is not None and existing is not value:
            raise RuntimeError(f"PEFT mapping {key!r} already registered to {existing!r}.")
        mapping[key] = value

def get_dense_model(model, config, adapter_name="default", low_cpu_mem_usage=False):
    register_dense_peft_method()
    # ★ peft 0.17.1 COMPAT: DensePeftModelForCausalLM init may differ slightly.
    # Tested against peft 0.17.1 BaseTuner.__init__ signature.
    return DensePeftModelForCausalLM(
        model, config, adapter_name=adapter_name,
        autocast_adapter_dtype=False, low_cpu_mem_usage=low_cpu_mem_usage,
    )

register_dense_peft_method()
```

### Step 8: Verification

- [ ] Run: `python -m py_compile run-lora/utils/dense_peft.py`
- [ ] Write a smoke test in a temporary script that:
  1. Creates a tiny `nn.Linear(4, 8)` wrapped in a `nn.Module`
  2. Calls `get_dense_model(model, DenseConfig(target_modules=["linear"]))`
  3. Checks that `.dense_delta["default"]` has shape `(8, 4)` and is all zeros
  4. Checks that forward pass produces same output as original (since ΔW=0)

---

## Task 2: Create CurvatureDense Variant

**Files:**
- Create: `run-lora/utils/curvature_dense.py`

**Interfaces:**
- Produces: `CurvatureDense(DenseVariant)` — bilateral projection variant
- Produces: `apply_curvature_dense(model)` — attaches CurvatureDense to all DenseLinear layers
- Produces: `get_curvature_dense_model(model, config)` — convenience constructor
- Produces: `CurvatureDenseTrainerMixin` — extends CurvatureTrainerMixin for Dense
- Produces: `CurvatureDenseTrainer(CurvatureDenseTrainerMixin, Trainer)` — concrete trainer

### Step 1: Implement `CurvatureDense(DenseVariant)`

```python
class CurvatureDense(DenseVariant):
    """Dense variant with bilateral low-curvature projection."""

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
    def get_delta_weight(module, adapter_name, delta):
        U_in_bar = getattr(module, f"U_in_bar_{adapter_name}").to(delta)
        U_out_bar = getattr(module, f"U_out_bar_{adapter_name}").to(delta)
        projected = delta
        if U_out_bar.shape[1] > 0:
            projected = projected - U_out_bar @ (U_out_bar.T @ projected)
        if U_in_bar.shape[1] > 0:
            projected = projected - (projected @ U_in_bar) @ U_in_bar.T
        return projected

    @staticmethod
    def forward(module, active_adapter, x, result, **kwargs):
        delta = module.dense_delta[active_adapter]
        U_in_bar = getattr(module, f"U_in_bar_{active_adapter}").to(delta)
        U_out_bar = getattr(module, f"U_out_bar_{active_adapter}").to(delta)

        # Token-level projection (right side: project input; left side: project output)
        x_projected = x.to(delta.dtype)
        if U_in_bar.shape[1] > 0:
            x_projected = x_projected - (x_projected @ U_in_bar) @ U_in_bar.T

        update = F.linear(x_projected, delta)
        if U_out_bar.shape[1] > 0:
            update = update - (update @ U_out_bar) @ U_out_bar.T

        return result + update.to(result.dtype)
```

> **Why token-level?** For full-rank ΔW, folding projections into the weight (parameter-level) costs O(d²·k) which dwarfs token-level O(B·T·d·k) for typical batch sizes. Unlike CurvatureLoRA's O(r·d·k) parameter-level folding, Dense's parameter-level approach is strictly less efficient.

### Step 2: Implement `apply_curvature_dense` and `get_curvature_dense_model`

```python
def _get_active_adapter(module):
    if hasattr(module, "active_adapters") and module.active_adapters:
        return module.active_adapters[0]
    if hasattr(module, "_active_adapter"):
        return module._active_adapter
    return None

def apply_curvature_dense(model, adapter_name=None):
    augmented = 0
    total_dense = 0
    for _, module in model.named_modules():
        if not isinstance(module, DenseLinear):
            continue
        total_dense += 1
        active = adapter_name or _get_active_adapter(module)
        if active is None or active not in module.dense_delta:
            continue
        module.dense_variant[active] = CurvatureDense()
        CurvatureDense.init(module, active)
        augmented += 1
    logger.info("CurvatureDense: augmented %s/%s Dense layers", augmented, total_dense)
    return model

def get_curvature_dense_model(model, config, adapter_name="default"):
    peft_model = get_dense_model(model, config, adapter_name=adapter_name)
    return apply_curvature_dense(peft_model, adapter_name=adapter_name)
```

### Step 3: Implement `CurvatureDenseTrainerMixin`

This is the **critical integration point** — it extends the existing `CurvatureTrainerMixin` from `curvature_lora.py` to work with `DenseLinear` modules instead of `LoraLayer` modules.

The key differences from the existing `CurvatureTrainerMixin`:
- `_setup_curvature()`: detects `dense_variant` instead of `lora_variant`
- `_apply_projections()`: writes to `dense_delta` reference instead of `lora_A`
- `_fold_projections_into_lora_weights()` → `_fold_projections_into_dense_weights()`: folds into `dense_delta` and removes `dense_variant` + buffers

```python
class CurvatureDenseTrainerMixin:
    """Mixin for CurvatureDense that reuses KFAC collection logic from CurvatureTrainerMixin."""

    def __init__(self, *args,
                 curvature_data_path: str,
                 curvature_energy_ratio: float = 0.9,
                 curvature_collection_batch_size: int = 1,
                 curvature_num_chunks: int = 1,
                 curvature_alpha_tr: float = 0.0,   # deprecated — kept for signature compat
                 curvature_delta_tr: float = 1e-12,  # deprecated — kept for signature compat
                 **kwargs):
        if not curvature_data_path:
            raise ValueError("curvature_data_path is required for CurvatureDense.")
        if not 0 < float(curvature_energy_ratio) <= 1:
            raise ValueError(f"curvature_energy_ratio must be in (0, 1], got {curvature_energy_ratio}.")
        if int(curvature_collection_batch_size) < 1:
            raise ValueError(f"curvature_collection_batch_size must be >= 1.")
        if int(curvature_num_chunks) < 1:
            raise ValueError(f"curvature_num_chunks must be >= 1.")

        self.curvature_data_path = curvature_data_path
        self.curvature_energy_ratio = float(curvature_energy_ratio)
        self.curvature_collection_batch_size = int(curvature_collection_batch_size)
        self.curvature_num_chunks = int(curvature_num_chunks)
        # trust-region is deprecated — store but don't use
        self.curvature_alpha_tr = 0.0
        self.curvature_delta_tr = 1e-12
        self._target_modules: Dict[str, torch.nn.Module] = {}
        self._adapter_names: Dict[str, str] = {}
        self._local_projections: Dict[str, tuple] = {}
        self._curvature_fisher_generator: Optional[torch.Generator] = None
        self.ref_model = None
        super().__init__(*args, **kwargs)
        self._fail_on_zero3()
        self.add_callback(CurvatureDenseCallback(self))
        self._setup_curvature()

    # ... (rest of implementation below)
```

### Step 4: Implement `_setup_curvature()` for Dense

```python
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
        raise ValueError("CurvatureDense trainer found no modules with attached curvature variants.")

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

# _build_curvature_loader, _tokenizer, _valid_target_mask, _compute_curvature_fisher_loss
# are IDENTICAL to the existing CurvatureTrainerMixin — import/reuse them.
```

### Step 5: Implement `_apply_projections()` for Dense

```python
def _apply_projections(self) -> None:
    """Write projection buffers to DenseLinear modules."""
    try:
        if self.args.world_size > 1:
            self._distributed_apply_projections()
        else:
            for name, module in self._target_modules.items():
                adapter_name = self._adapter_names[name]
                U_in_bar, U_out_bar = self._local_projections[name]
                # ★ Reference for dtype/device: dense_delta instead of lora_A
                ref = module.dense_delta[adapter_name]
                module._buffers[f"U_in_bar_{adapter_name}"] = U_in_bar.to(ref)
                module._buffers[f"U_out_bar_{adapter_name}"] = U_out_bar.to(ref)
    except Exception:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        raise
```

### Step 6: Implement `_distributed_apply_projections()` for Dense

Same as the LoRA version but uses `module.dense_delta[adapter_name]` as the `ref` tensor instead of `module.lora_A[adapter_name].weight`.

### Step 7: Implement `_fold_projections_into_dense_weights()`

```python
def _fold_projections_into_dense_weights(self) -> None:
    """Fold projection buffers into dense_delta, then remove CurvatureDense variant.
    
    After folding, the module becomes a plain Dense adapter with the projection
    baked into ΔW. This ensures saved checkpoints work with standard Dense loading.
    """
    logger.info("CurvatureDense: folding projection buffers into dense_delta before saving.")
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
                raise RuntimeError(f"Refusing to fold empty CurvatureDense projections for {name}.")

            # Apply bilateral projection to delta: (I-U_outU_out^T) @ delta @ (I-U_inU_in^T)
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

    logger.info("CurvatureDense: folded %d/%d Dense layers.", folded, len(self._target_modules))
```

### Step 8: Implement `CurvatureDenseCallback`

Same as `CurvatureCallback` but calls `curvature_on_train_begin` and `curvature_on_train_end` — **no `on_optimizer_step` hook** since trust-region is deprecated.

```python
class CurvatureDenseCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer.curvature_on_train_begin()

    def on_train_end(self, args, state, control, **kwargs):
        self.trainer.curvature_on_train_end()
```

### Step 9: Implement `curvature_on_train_begin` and `curvature_on_train_end`

```python
def curvature_on_train_begin(self):
    logger.info("CurvatureDense: collecting static KFAC projections from %s", self.curvature_data_path)
    self._collect_factors()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    self._apply_projections()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def curvature_on_train_end(self):
    self._tracker.remove_hooks()
    self._fold_projections_into_dense_weights()
```

### Step 10: Reuse `_collect_factors` from CurvatureTrainerMixin

The `_collect_factors()` method from the existing `CurvatureTrainerMixin` is **identical** for both LoRA and Dense — it uses `KFACTracker` hooks on the base layer, not the adapter layer. The `_curvature_loader`, `_compute_curvature_fisher_loss`, `_DisabledAdapterModel`, and `_tokenizer` methods are also identical.

**Strategy**: Factor out the shared methods into a base class or directly inherit-and-reuse. The cleanest approach for this repo is to have `CurvatureDenseTrainerMixin` contain its own copies of these methods (they are not huge and this avoids fragile inheritance chains).

> **Note on `_DisabledAdapterModel`**: For CurvatureLoRA, this uses `model.disable_adapter()` (a PEFT context manager that temporarily disables LoRA). For Dense, the equivalent disable mechanism works through `DenseLayer._disable_adapters`. However, since KFAC collection hooks are attached to the **base layer** (the original `nn.Linear` inside `DenseLinear`), and the hooks capture activations/gradients on the base layer only, the adapter does not interfere. We still use `_DisabledAdapterModel` if the retain loss is Fisher (KL), to get reference logits without adapter influence.

### Step 11: Implement `CurvatureDenseTrainer`

```python
class CurvatureDenseTrainer(CurvatureDenseTrainerMixin, Trainer):
    pass
```

### Step 12: Verification

- [ ] Run: `python -m py_compile run-lora/utils/curvature_dense.py`
- [ ] Write a smoke test:
  1. Create tiny model + `get_curvature_dense_model()`
  2. Verify `dense_variant["default"]` is a `CurvatureDense` instance
  3. Verify `U_in_bar_default` and `U_out_bar_default` buffers exist with shape `(d, 0)`
  4. Set a non-trivial `dense_delta` and non-empty projection buffers, verify forward matches `(I-U_outU_out^T) @ ΔW @ (I-U_inU_in^T)`

---

## Task 3: Integrate into `train.py`

**Files:**
- Modify: `run-lora/train.py`

**Interfaces:**
- Consumes: `CurvatureDenseTrainer`, `apply_curvature_dense`, `get_curvature_dense_model` from `curvature_dense.py`
- Produces: `--use_curvature_dense` flag
- Produces: Dense adapter construction in `build_model()`

### Step 1: Add dataclass fields

In `TrainingArguments` dataclass, add after the existing `use_curvature_lora` block:

```python
## CurvatureDense
use_curvature_dense: Optional[bool] = field(default=False)
# Reuses the same curvature_data_path, curvature_energy_ratio,
# curvature_collection_batch_size, curvature_num_chunks from CurvatureLoRA.
```

### Step 2: Add import

```python
from utils.curvature_dense import CurvatureDenseTrainer, apply_curvature_dense, get_curvature_dense_model
from utils.dense_peft import DenseConfig, get_dense_model
```

### Step 3: Add model construction branch in `build_model()`

After the existing `if script_args.use_curvature_lora:` block, add:

```python
if script_args.use_curvature_dense:
    if resume_from_checkpoint_dir is not None:
        raise ValueError("CurvatureDense does not support resuming from unfinished checkpoints.")
    if script_args.curvature_data_path is None:
        raise ValueError("--curvature_data_path is required when --use_curvature_dense True.")
    if str(script_args.save_strategy).lower().split(".")[-1] != "no":
        raise ValueError("CurvatureDense requires --save_strategy no; only the final folded model is saved.")
    # Dense does not use lora_rank/lora_alpha — config is target_modules only
```

And in the PEFT construction block (inside `if not script_args.full_finetune:`), after `model = get_peft_model(model, peft_config)` and after the `if script_args.use_curvature_lora:` block:

```python
if script_args.use_curvature_dense:
    # Construct Dense adapter then attach CurvatureDense variant
    dense_config = DenseConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=script_args.target_modules.split(','),
    )
    model = get_curvature_dense_model(model, dense_config)
```

> **Important**: `get_curvature_dense_model` internally calls `get_dense_model` which replaces `nn.Linear` with `DenseLinear`, then `apply_curvature_dense` which attaches `CurvatureDense`. This means we should NOT call `get_peft_model` with a `LoraConfig` first — Dense is mutually exclusive with LoRA.

Add mutual exclusion check:

```python
if script_args.use_curvature_lora and script_args.use_curvature_dense:
    raise ValueError("Cannot use both CurvatureLoRA and CurvatureDense simultaneously.")
```

### Step 4: Add trainer dispatch

In the `train()` function, after the existing `if script_args.use_curvature_lora:` trainer creation block:

```python
elif script_args.use_curvature_dense:
    if script_args.use_loft:
        raise ValueError("CurvatureDense cannot be combined with LoFT optimizer mode.")
    trainer = CurvatureDenseTrainer(
        model=model,
        tokenizer=tokenizer,
        args=script_args,
        curvature_data_path=script_args.curvature_data_path,
        curvature_energy_ratio=script_args.curvature_energy_ratio,
        curvature_collection_batch_size=script_args.curvature_collection_batch_size,
        curvature_num_chunks=script_args.curvature_num_chunks,
        curvature_alpha_tr=0.0,   # deprecated
        curvature_delta_tr=1e-12, # deprecated
        **data_module,
    )
```

### Step 5: Verification

- [ ] Run: `python -m py_compile run-lora/train.py`
- [ ] Check that `--help` shows `--use_curvature_dense`

---

## Task 4: Wire Shell Scripts

**Files:**
- Modify: `run-lora/scripts/train.sh`
- Modify: `run-lora/scripts/test.sh`
- Modify: `run-lora/scripts/run_best_common.sh`
- Modify: `run-lora/scripts/qwen/math_curvature_dense.sh`

### Step 1: Update `train.sh`

#### a. Add `CurvatureDense` to the allowed PEFT regex:
```bash
if [[ ! "$peft" =~ ^(LoRA|CurvatureLoRA|CurvatureDense|Corda-KPM|Corda-IPM|PiSSA|MiLoRA|DoRA|OLoRA|InitAB|LoRA-GA|GraLoRA|RandLoRA|LoFT)$ ]]; then
```

#### b. Add `CurvatureDense` dispatch branch (after the `CurvatureLoRA` branch):

```bash
elif [[ "$peft" == "CurvatureDense" ]]; then
  deepspeed --master_port=$master_port --include=localhost:"$gpus" train.py \
    --use_curvature_dense True \
    --curvature_data_path "$curvature_data_path" \
    --curvature_energy_ratio "$curvature_energy_ratio" \
    --curvature_collection_batch_size 1 \
    --curvature_num_chunks "$curvature_num_chunks" \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path "$model" \
    --full_finetune False \
    --bf16 \
    --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --data_path "$data" \
    --sub_task "$task" \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir "$output_path" \
    --num_train_epochs "$num_train_epochs" \
    --model_max_length "$model_max_length" \
    --per_device_train_batch_size "$per_bs" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --learning_rate "$lr" \
    --save_strategy "no" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "$curvature_lr_scheduler" \
    --report_to "wandb" \
    --trial_id $trial_id \
    --logging_steps $logging_steps \
    --gradient_checkpointing True
```

Key differences from CurvatureLoRA:
- **No** `--lora_rank`, `--lora_alpha`, `--lora_dropout`, `--init_weights` (Dense is not LoRA)
- **No** `--curvature_alpha_tr`, `--curvature_delta_tr` (trust-region deprecated)

### Step 2: Update `test.sh`

#### a. Add `CurvatureDense` to validation regex and help text:
```bash
if [[ ! "$peft" =~ ^(LoRA|CurvatureLoRA|CurvatureDense|Corda-KPM|Corda-IPM|PiSSA|MiLoRA|DoRA|OLoRA|InitAB|LoRA-GA|GraLoRA|RandLoRA|LoFT)$ ]]; then
```

#### b. Add `CurvatureDense` to the base_model resolution:
```bash
else
  # LoRA, CurvatureLoRA, CurvatureDense, DoRA, GraLoRA, LoFT — base model with standard adapter
  base_model="$model"
fi
```

**CRITICAL**: For CurvatureDense, the saved model is a `DensePeftModelForCausalLM.save_pretrained()` output — a **merged full model**, not an adapter. However, our folding step (`_fold_projections_into_dense_weights`) converts it into a standard Dense adapter, and `DensePeftModelForCausalLM.save_pretrained()` exports a full merged model anyway. For vLLM inference, we need to check whether vLLM supports loading from the merged model directly.

**vLLM compatibility**: After `curvature_on_train_end` folds projections and `DensePeftModelForCausalLM.save_pretrained()` saves as a merged full model, the `adapter_model` directory contains a standard HuggingFace model. The test script should use this as the base model directly:

```bash
elif [[ "$peft" == "CurvatureDense" ]]; then
  # CurvatureDense saves a merged full model — load it directly without --lora
  base_model="$output_path/adapter_model"
  lora_path=""   # clear lora_path — no separate adapter
```

This requires adjusting the vLLM invocation in test.sh to conditionally omit `--lora`:

```bash
# Build optional --lora argument
lora_args=()
if [[ -n "$lora_path" ]]; then
  lora_args=(--lora "$lora_path")
fi
```

When `lora_path` is empty, `lora_args` is empty and vLLM loads the model directly.

### Step 3: Update `run_best_common.sh`

The existing script already passes `curvature_*` arguments to `train.sh` via the shared call. No changes needed beyond ensuring the output path naming uses `CurvatureDense`:

In the `output_path` construction:
```bash
output_path=$(readlink -m \
  "${output_home}/output/${task}-${peft}-${model_abbr}-r${rank}/bs${bs}-lr${lr}-trial${seed}${_er_suffix}")
```

This automatically generates paths like `metamath-CurvatureDense-qwen-3-0.6b-r128/...` — no change needed.

### Step 4: Update `qwen/math_curvature_dense.sh`

The existing file already has the correct structure. Verify:
- `peft_methods=("CurvatureDense")` ✓
- `curvature_energy_ratio=0.7` ✓
- `peft_lrs["CurvatureDense"]` ✓
- Remove `rank` if not needed for CurvatureDense (but keep for path naming consistency)

### Step 5: Verification

- [ ] Run: `bash -n run-lora/scripts/train.sh`
- [ ] Run: `bash -n run-lora/scripts/test.sh`
- [ ] Run: `bash -n run-lora/scripts/run_best_common.sh`
- [ ] Run: `bash -n run-lora/scripts/qwen/math_curvature_dense.sh`

---

## Task 5: Unit Tests

**Files:**
- Create: `run-lora/utils/test_dense_peft.py`

### Step 1: Write tests for Dense PEFT infrastructure

```python
import torch
import torch.nn as nn
import unittest

from dense_peft import (
    DenseConfig, DenseLinear, DenseLayer, DenseVariant,
    get_dense_model, register_dense_peft_method,
)

class TrivialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 8)

    def forward(self, x):
        return self.linear(x)


class TestDensePEFT(unittest.TestCase):

    def setUp(self):
        register_dense_peft_method()

    def test_dense_linear_creation(self):
        model = TrivialModel()
        config = DenseConfig(target_modules=["linear"], task_type="CAUSAL_LM")
        peft_model = get_dense_model(model, config)
        dense_module = peft_model.base_model.model.linear
        self.assertIsInstance(dense_module, DenseLinear)
        self.assertIn("default", dense_module.dense_delta)
        self.assertEqual(dense_module.dense_delta["default"].shape, (8, 4))

    def test_dense_zero_init_forward(self):
        """With ΔW=0, output matches base model."""
        model = TrivialModel()
        config = DenseConfig(target_modules=["linear"], task_type="CAUSAL_LM")
        peft_model = get_dense_model(model, config)
        x = torch.randn(2, 4)
        out_base = model(x)
        out_dense = peft_model(x)
        self.assertTrue(torch.allclose(out_base, out_dense, atol=1e-6))

    def test_dense_merge_unmerge_roundtrip(self):
        model = TrivialModel()
        config = DenseConfig(target_modules=["linear"], task_type="CAUSAL_LM")
        peft_model = get_dense_model(model, config)
        dense_module = peft_model.base_model.model.linear

        # Set a non-zero delta
        delta = torch.randn(8, 4)
        dense_module.dense_delta["default"].data.copy_(delta)

        x = torch.randn(2, 4)
        out_with_delta = peft_model(x)

        # Merge
        dense_module.merge()
        self.assertTrue(dense_module.merged)
        out_merged = peft_model(x)
        self.assertTrue(torch.allclose(out_with_delta, out_merged, atol=1e-5))

        # Unmerge
        dense_module.unmerge()
        out_unmerged = peft_model(x)
        self.assertTrue(torch.allclose(out_unmerged, out_base, atol=1e-6))

    def test_dense_model_state_dict_keys(self):
        model = TrivialModel()
        config = DenseConfig(target_modules=["linear"], task_type="CAUSAL_LM")
        peft_model = get_dense_model(model, config)
        state_keys = set(peft_model.state_dict().keys())
        # Should contain dense_delta
        self.assertTrue(any("dense_delta" in k for k in state_keys))
```

### Step 2: Write tests for CurvatureDense

```python
from curvature_dense import CurvatureDense, apply_curvature_dense
from dense_peft import get_dense_model, DenseConfig


class TestCurvatureDense(unittest.TestCase):

    def setUp(self):
        register_dense_peft_method()

    def test_curvature_dense_buffers_created(self):
        model = TrivialModel()
        config = DenseConfig(target_modules=["linear"], task_type="CAUSAL_LM")
        peft_model = get_dense_model(model, config)
        apply_curvature_dense(peft_model)
        dense_module = peft_model.base_model.model.linear
        self.assertIn("default", dense_module.dense_variant)
        self.assertIsInstance(dense_module.dense_variant["default"], CurvatureDense)
        self.assertTrue(hasattr(dense_module, "U_in_bar_default"))
        self.assertTrue(hasattr(dense_module, "U_out_bar_default"))

    def test_curvature_dense_forward_matches_manual_projection(self):
        model = TrivialModel()
        config = DenseConfig(target_modules=["linear"], task_type="CAUSAL_LM")
        peft_model = get_dense_model(model, config)
        apply_curvature_dense(peft_model)
        dense_module = peft_model.base_model.model.linear

        # Set non-trivial delta and projection bases
        delta = torch.randn(8, 4)
        dense_module.dense_delta["default"].data.copy_(delta)

        # Create a simple projection basis (orthonormal)
        U_in = torch.eye(4)[:, :2]   # project out first 2 input dims
        U_out = torch.eye(8)[:, :2]  # project out first 2 output dims
        dense_module._buffers["U_in_bar_default"] = U_in.to(delta)
        dense_module._buffers["U_out_bar_default"] = U_out.to(delta)

        x = torch.randn(2, 4)
        out_model = peft_model(x)

        # Manual computation
        x_proj = x - (x @ U_in) @ U_in.T
        manual_update = F.linear(x_proj, delta)
        manual_update = manual_update - (manual_update @ U_out) @ U_out.T
        expected = F.linear(x, model.linear.weight) + manual_update

        self.assertTrue(torch.allclose(out_model, expected, atol=1e-5))
```

### Step 3: Verification

- [ ] Run: `cd run-lora/utils && python -m unittest test_dense_peft.py -v`
- [ ] All tests pass.

---

## Task 6: End-to-End Smoke Test

**Goal:** Validate the full pipeline with a tiny model.

### Step 1: Create minimal smoke test script

```bash
# run-lora/scripts/smoke_curvature_dense.sh
python -c "
import torch
import transformers
from utils.dense_peft import DenseConfig, get_dense_model, register_dense_peft_method
from utils.curvature_dense import apply_curvature_dense, get_curvature_dense_model

register_dense_peft_method()

# Create a tiny test model
config = transformers.AutoConfig.from_pretrained('HuggingFaceTB/SmolLM-135M', trust_remote_code=True)
# ... (construct tiny model with 1 layer)
"
```

### Step 2: Run smoke test

- [ ] The pipeline completes without errors.
- [ ] KFAC collection produces non-empty projection bases.
- [ ] Training runs for 1 step without OOM.
- [ ] `curvature_on_train_end` folds projections successfully.
- [ ] Saved model can be loaded with `AutoModelForCausalLM.from_pretrained()`.

---

## Summary of All Files

| File | Action | Purpose |
|---|---|---|
| `run-lora/utils/dense_peft.py` | **CREATE** | Dense PEFT infrastructure (DenseVariant, DenseLayer, DenseLinear, DenseConfig, DenseModel, DensePeftModelForCausalLM) |
| `run-lora/utils/curvature_dense.py` | **CREATE** | CurvatureDense variant + CurvatureDenseTrainerMixin + CurvatureDenseTrainer |
| `run-lora/utils/test_dense_peft.py` | **CREATE** | Unit tests for Dense PEFT and CurvatureDense |
| `run-lora/train.py` | **MODIFY** | Add `--use_curvature_dense` flag, Dense model construction, trainer dispatch |
| `run-lora/scripts/train.sh` | **MODIFY** | Add `CurvatureDense` to PEFT regex, add dispatch branch |
| `run-lora/scripts/test.sh` | **MODIFY** | Add `CurvatureDense` to PEFT regex, handle merged-model loading for vLLM |
| `run-lora/scripts/run_best_common.sh` | **MODIFY** | Minor: ensure `CurvatureDense` naming in output paths |
| `run-lora/scripts/qwen/math_curvature_dense.sh` | **VERIFY** | Already configured correctly |

## Compatibility Checklist

| Check | Status |
|---|---|
| `BaseTuner._create_and_replace` signature | ✅ Uses `current_key` + `**optional_kwargs` |
| `BaseTunerLayer.set_adapter` with `inference_mode` | ✅ Try/except TypeError fallback |
| `PeftConfig.inference_mode` field | ✅ `getattr(dense_config, 'inference_mode', False)` |
| `active_adapters` (list) vs `active_adapter` (str) | ✅ Uses `active_adapters` list |
| `DenseModel._replace_module` 4-arg call | ✅ Compatible |
| Trust-region deprecated | ✅ `curvature_alpha_tr` hardcoded to 0.0 |
| Static KFAC collection only | ✅ No `on_epoch_begin` / `on_step_begin` refresh hooks |
| Training-end folding | ✅ `_fold_projections_into_dense_weights()` |
| vLLM inference compatibility | ✅ Saves as merged full model, loaded directly |
| DeepSpeed ZeRO-2 only | ✅ `_fail_on_zero3()` check |
