# CurvatureDense Implementation Handoff

> Session date: 2026-07-25 to 2026-07-27
> Goal: Add CurvatureDense — a full-rank additive PEFT adapter with KFAC bilateral curvature projection — to the lr-matters-lora experiment pipeline.

---

## What Was Built

CurvatureDense replaces the low-rank LoRA adapter ($\Delta W = BA$) with a full-rank additive adapter ($\Delta W \in \mathbb{R}^{d_{out} \times d_{in}}$), applying the same KFAC-based curvature projection as CurvatureLoRA but without the low-rank constraint. Unlike CurvatureLoRA, trust-region projection (`alpha_tr`) is deprecated.

### New Files

| File | Purpose |
|---|---|
| `run-lora/utils/dense_peft.py` | Full Dense PEFT infrastructure: `DenseVariant`, `DenseLayer`, `DenseLinear`, `DenseConfig`, `DenseModel`, `DensePeftModelForCausalLM`, `get_dense_model()` |
| `run-lora/utils/curvature_dense.py` | `CurvatureDense(DenseVariant)`, `CurvatureDenseTrainerMixin`, `CurvatureDenseTrainer`, `_DisabledDenseAdapterModel` |
| `run-lora/utils/test_dense_peft.py` | 15 unit tests covering Dense PEFT and CurvatureDense |
| `run-lora/scripts/llama/math_curvature_dense.sh` | Llama-2-7B math experiment (rank=32, bs=16, 2 GPUs) |
| `run-lora/scripts/llama/code_curvature_dense.sh` | Llama-2-7B code experiment (rank=32, bs=16, 2 GPUs) |
| `run-lora/scripts/qwen/math_curvature_dense.sh` | Qwen3-0.6B math experiment (rank=128, bs=64, 1 GPU) |
| `run-lora/scripts/gemma/math_curvature_dense.sh` | Gemma-3-1B math experiment (rank=128, bs=64, 1 GPU) |

### Modified Files

| File | Changes |
|---|---|
| `run-lora/train.py` | Added `--use_curvature_dense` flag, Dense model construction (skips LoRA), trainer dispatch, mutual exclusion guard |
| `run-lora/scripts/train.sh` | Added `CurvatureDense` PEFT name, DeepSpeed dispatch branch |
| `run-lora/scripts/test.sh` | Added `CurvatureDense` PEFT name, merged-model inference path |
| `run-lora/scripts/run_best_common.sh` | Energy-ratio suffix for both CurvatureLoRA and CurvatureDense |

---

## Critical peft 0.17.1 Compatibility Lessons

The original implementation plan was based on **peft 0.19.1** — the reference repo `hessian_lora`. The target repo uses **peft 0.17.1**. These versions differ substantially in what `BaseTuner` provides.

### DenseModel Must Implement 9 Methods (Not 6)

`BaseTuner` in peft 0.17.1 is a **skeleton**: it only provides `inject_adapter` (the dispatch loop) and `__init__`. Everything else must be implemented by the subclass. The 9 required methods:

| # | Method | Abstract? | Why Needed |
|---|---|---|---|
| 1 | `_prepare_adapter_config` | ✅ | Auto-infer target_modules |
| 2 | `_check_target_module_exists` | ✅ | Module matching (delegates to `peft.tuners.tuners_utils.check_target_module_exists`) |
| 3 | `_create_and_replace` | ✅ | Core module replacement logic |
| 4 | `_mark_only_adapters_as_trainable` | ✅ | Freeze all but `dense_delta` |
| 5 | `disable_adapter_layers` | ✅ | Must call `module.enable_adapters(False)` — NOT `module.disable_adapters = True` |
| 6 | `enable_adapter_layers` | ✅ | Must call `module.enable_adapters(True)` |
| 7 | `set_adapter` (model-level) | ❌ | Called by `inject_adapter` **during construction** — missing = instant `AttributeError` |
| 8 | `_replace_module` | ❌ | Not on `BaseTuner` in 0.17.1; `LoraModel` defines its own. Missing = crash at first module replacement |
| 9 | `__getattr__` | ❌ | `PeftModelForCausalLM.__init__` accesses `prepare_inputs_for_generation` through `self.base_model` during construction |

Methods 7-9 are the most dangerous: Python won't warn at class definition — they only fail at runtime, and 7 & 9 fail during **construction** (not training).

### `inference_mode` Must NOT Be Passed Down

In peft 0.17.1, `BaseTunerLayer.set_adapter()` signature is `(self, adapter_names: str | list[str])` — **no `inference_mode` kwarg**. The plan incorrectly passed `inference_mode` through the chain: `_create_and_replace` → `_create_new_module` → `DenseLinear.__init__` → `update_layer` → `set_adapter(..., inference_mode=...)`. This was entirely removed — `inject_adapter` handles `inference_mode` internally.

### `disable_adapters` Is a Read-Only Property

`BaseTunerLayer.disable_adapters` is a property (returns `self._disable_adapters`). You cannot set it directly. Use `module.enable_adapters(enabled=False)` instead.

### `register_dense_peft_method` Must Be Truly Idempotent

When the same module is imported via different paths (`from dense_peft import ...` vs `from utils.dense_peft import ...`), Python creates different class objects. An `is`-based identity check fails. Solution: unconditionally overwrite the PEFT dispatch maps (same logical value, harmless overwrite).

---

## Other Pitfalls Encountered

### 1. Import Path Resolution (`curvature_dense.py`)

**Problem**: `curvature_dense.py` lives in `utils/` alongside `curvature_lora.py`. When imported from `train.py` (in `run-lora/`), `from utils.curvature_lora import ...` works. When imported from a test in `utils/`, only `from curvature_lora import ...` works.

**Solution**: At module load time, compute the parent directory and insert it into `sys.path`:
```python
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
```
Then always use `from utils.xxx` imports.

### 2. `_DisabledAdapterModel` Doesn't Work on Raw DenseModel

**Problem**: `curvature_lora._DisabledAdapterModel` uses `model.disable_adapter()` — a `PeftModel` context manager. `DenseModel` (a raw `BaseTuner` subclass) doesn't have this.

**Solution**: Created `_DisabledDenseAdapterModel` in `curvature_dense.py` that uses `disable_adapter_layers()`/`enable_adapter_layers()` (the `BaseTuner` abstract methods that `DenseModel` implements).

### 3. `_merged_base_state_dict` Key Naming

**Problem**: `named_modules()` returns module names with the `DenseModel.model.` prefix (e.g., `"model.linear"`), but the merged state dict stores keys without that prefix (`"linear.weight"`). The delta key lookup also had a double-prefix bug (`f"{root}{_name}.dense_delta..."` where `root` and `_name` both contained `"model."`).

**Solution**: Strip `root` from `_name` to get `rel_name`, use `f"{_name}.dense_delta..."` (no double root) for source lookup, store under `f"{rel_name}.weight"`.

### 4. `nn.Linear` Bias in Test Expected Values

**Problem**: Tests used `F.linear(x, weight)` to compute expected output, which omits bias. But `DenseLayer.forward()` calls `base_layer(x)` which includes bias.

**Solution**: Use `dense_module.get_base_layer()(x)` for all test expected-value computations.

### 5. `model(x)` After In-Place Replacement

**Problem**: After `get_dense_model(model, config)`, `model.linear` is replaced in-place with `DenseLinear` — so `model(x)` post-wrap already includes the delta. Using it as "base output" reference is wrong.

**Solution**: Use `dense_module.get_base_layer()(x)` for "no adapter" reference output.

### 6. `remove_unused_columns` Drops All Columns

**Problem**: `DenseModel.forward` inherits `*args, **kwargs` from `BaseTuner`. Trainer's `_remove_unused_columns` can't match any dataset column against this generic signature, so it drops everything.

**Solution**: Set `self.args.remove_unused_columns = False` in `CurvatureDenseTrainerMixin.__init__`.

### 7. `print_trainable_parameters` Not Available on DenseModel

**Problem**: `PeftModel.print_trainable_parameters()` doesn't exist on raw `DenseModel`.

**Solution**: Skip the call when `--use_curvature_dense True` (purely cosmetic log, zero training impact).

### 8. Trust-Region Deprecated

The plan included `curvature_alpha_tr`/`curvature_delta_tr` trust-region projection. This was explicitly deprecated per user request. `CurvatureDenseCallback` has no `on_optimizer_step` hook, `alpha_tr` is hardcoded to 0.0.

---

## Key Architectural Decisions

### Token-Level vs Parameter-Level Projection

CurvatureLoRA uses **parameter-level** folding ($O(r \cdot d \cdot k)$) because it's faster for small ranks. CurvatureDense uses **token-level** projection because for full-rank ΔW, parameter-level folding is $O(d^2 \cdot k)$ which dwarfs token-level $O(B \cdot T \cdot d \cdot k)$.

### Static KFAC Collection Only

No refresh/EMA — KFAC factors are collected once at `on_train_begin`. No `on_epoch_begin`/`on_step_begin` refresh hooks.

### Full Model Save (Not Adapter)

Since ΔW has the same shape as base weights, there's no storage benefit to saving just the adapter. `DensePeftModelForCausalLM` saves a fully merged HuggingFace model. The test script loads it directly without `--lora`.

### CurvatureLoRA Mutual Exclusion

CurvatureDense is mutually exclusive with CurvatureLoRA — cannot use both simultaneously.

---

## Experiment Script Hyperparameters

All scripts use `curvature_energy_ratio=0.7`, `curvature_lr_scheduler=cosine`, `peft_lrs["CurvatureDense"]="1.1247e-4 2.0000e-4 3.5566e-4 6.3246e-4"`.

| Script | Model | rank | bs | per_bs | num_chunks | GPUs |
|---|---|---|---|---|---|---|
| `qwen/math_curvature_dense.sh` | Qwen3-0.6B | 128 | 64 | 4 | 1 | 1 |
| `llama/math_curvature_dense.sh` | Llama-2-7B | 32 | 16 | 1 | 4 | 2 |
| `llama/code_curvature_dense.sh` | Llama-2-7B | 32 | 16 | 4 | 2 | 2 |
| `gemma/math_curvature_dense.sh` | Gemma-3-1B | 128 | 64 | 4 | 1 | 1 |
