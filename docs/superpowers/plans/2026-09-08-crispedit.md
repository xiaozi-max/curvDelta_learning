# CrispEdit Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate CrispEdit as a fixed-cache, full-model baseline for all configured transformer projection layers in `run-lora`.

**Architecture:** Add a focused CrispEdit utility module containing target discovery, one-pass chunked K-FAC collection, Kronecker eigen-mask construction, and an AdamW-compatible gradient projection wrapper. Add a dedicated Trainer/model-save path in `train.py`, then expose it through the unified shell and evaluation loaders. Reuse the existing SFT data pipeline and FineFineWeb/KFAC token handling.

**Tech Stack:** Python 3.12, PyTorch, Transformers Trainer, Accelerate/DeepSpeed, Hugging Face Datasets, existing `run-lora/utils/curvature_lora.py` KFAC tracker, AdamW, cosine scheduler.

## Global Constraints

- CrispEdit directly updates base-model weights; it does not create a LoRA or Dense adapter.
- Default target modules are `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` across all transformer layers.
- K-FAC cache is collected exactly once from FineFineWeb context and reused unchanged.
- Projection must implement `U_B @ ((U_B.T @ grad @ U_A) * M.T) @ U_A.T` for every target 2-D weight gradient.
- Optimizer is `torch.optim.AdamW`; scheduler is always `--lr_scheduler_type cosine`.
- Do not implement cache reset, cache recomputation, sequential cache merging, or momentum reprojection.
- Preserve unrelated dirty-worktree changes.

## Reference Files For Reproduction

Read these files in addition to this plan: `run-lora/train.py`, `run-lora/scripts/train.sh`, `run-lora/scripts/test.sh`, `run-lora/utils/eval_world_knowledge.py`, `run-lora/utils/save_finefineweb.py`, `run-lora/utils/curvature_lora.py`, the new CrispEdit utility/trainer modules, and the original CrispEdit references `F:/project/CrispEdit/crispedit.py`, `F:/project/CrispEdit/easyeditor/models/crispedit/projected_adam.py`, `F:/project/CrispEdit/easyeditor/models/crispedit/utils.py`, and `F:/project/CrispEdit/easyeditor/models/rome/layer_stats.py`.

### Task 1: Add pure projection and cache utilities

**Files:**
- Create: `run-lora/utils/crispedit.py`
- Test: `run-lora/utils/test_crispedit.py`

**Interfaces:**
- Produce `discover_target_modules(model, target_modules) -> dict[str, torch.nn.Linear]`.
- Produce `build_kronecker_projection_cache(factor_a, factor_b, energy_ratio) -> dict[str, torch.Tensor]` with `Ua`, `Ub`, and boolean `M`.
- Produce `project_gradient(grad, cache) -> torch.Tensor` using the exact CrispEdit orientation and `M.T`.
- Produce `projected_adamw(params, projection_cache, **adamw_kwargs) -> torch.optim.Optimizer` or an equivalent optimizer class whose `step()` projects only mapped 2-D gradients before delegating to AdamW.

- [ ] Write tests with small SPD factors and a hand-computed expected projected gradient; assert `M.T` orientation, untouched non-2-D gradients, and unmapped parameters.
- [ ] Run `python -m pytest run-lora/utils/test_crispedit.py -q` and confirm the pre-implementation test fails.
- [ ] Implement eigendecomposition, energy thresholding, cache storage, and AdamW projection without any reset/recompute API.
- [ ] Run the focused test and confirm it passes.

### Task 2: Implement one-pass chunked FineFineWeb K-FAC collection

**Files:**
- Modify: `run-lora/utils/crispedit.py`
- Modify: `run-lora/utils/curvature_lora.py` only where a small reusable hook/data-collator seam is required
- Test: `run-lora/utils/test_crispedit.py`

**Interfaces:**
- Produce `collect_fixed_kfac_cache(model, tokenizer, data_path, target_modules, batch_size, num_chunks, max_length, energy_ratio) -> dict[str, dict]`.
- The returned map contains CPU-safe `Ua`, `Ub`, and `M` for every target module.

- [ ] Add a deterministic tiny-linear-model test comparing one-chunk and multi-chunk factor/cache shapes and values within tolerance.
- [ ] Run the test to verify the new collector is initially absent or failing.
- [ ] Reuse the existing valid-token mask semantics from `KFACTracker`; collect each chunk once, normalize by token count, eigendecompose, move final caches to CPU, and release hooks/transient tensors before the next chunk.
- [ ] Run the collector tests and confirm one- versus multi-chunk outputs agree.

### Task 3: Add CrispEdit model/trainer lifecycle to `train.py`

**Files:**
- Modify: `run-lora/train.py`
- Modify: `run-lora/utils/crispedit.py`
- Test: `run-lora/utils/test_crispedit.py`

**Interfaces:**
- Add `use_crispedit`, `crispedit_data_path`, `crispedit_energy_ratio`, `crispedit_collection_batch_size`, `crispedit_num_chunks`, and a configurable projection target-module string to `TrainingArguments`.
- Add a trainer path that freezes all parameters except discovered target weights, collects the fixed cache before training, constructs AdamW, and attaches the cosine scheduler through standard Trainer arguments.
- Save a complete model and tokenizer under `<output_dir>/adapter_model`, plus `completed` marker.

- [ ] Add a model-construction test asserting only all requested projection weights require gradients.
- [ ] Run it before implementation and record the expected failure.
- [ ] Implement validation against incompatible PEFT flags, require calibration data, call fixed collection once before the first optimizer step, and use the existing SFT dataset/collator unchanged.
- [ ] Ensure the trainer never invokes cache reset/recompute and never projects optimizer momentum separately.
- [ ] Run focused lifecycle tests with a tiny causal-LM stub and verify AdamW/cosine scheduler objects.

### Task 4: Expose training CLI and reproducible model-family scripts

**Files:**
- Modify: `run-lora/scripts/train.sh`
- Create or modify: `run-lora/scripts/qwen/math_crispedit.sh`, `run-lora/scripts/llama/math_crispedit.sh`, `run-lora/scripts/gemma/math_crispedit.sh`
- Modify: `run-lora/README.md`

**Interfaces:**
- `--peft CrispEdit` selects the branch, accepts calibration path/energy/chunk options, and always emits `--lr_scheduler_type cosine`.
- Scripts document FineFineWeb JSONL preparation, target modules, output path, and full-model artifact location.

- [ ] Add shell dispatch/help text and validation for CrispEdit.
- [ ] Add command construction using the dedicated `--use_crispedit` path and full-model save semantics while retaining only target weights trainable; do not set `--full_finetune` unless the implementation explicitly separates that flag from target freezing.
- [ ] Add explicit `--lr_scheduler_type "cosine"`; do not route through `curvature_lr_scheduler`.
- [ ] Run `bash -n` on changed scripts and `bash ./scripts/train.sh --help` from `run-lora`.

### Task 5: Integrate full-model evaluation and documentation

**Files:**
- Modify: `run-lora/scripts/test.sh`
- Modify: `run-lora/utils/eval_world_knowledge.py`
- Modify: `run-lora/scripts/run_best_common.sh` if method dispatch requires it
- Modify: `run-lora/README.md`

**Interfaces:**
- CrispEdit loads `<output_dir>/adapter_model` as a complete Transformers model, with no adapter config.
- Task generation/scoring and world-knowledge evaluation produce the same files and metrics as FFT/CurvatureDense.

- [ ] Add loader-path tests using a tiny saved full model and assert no PEFT wrapping occurs.
- [ ] Run the tests before changing dispatch and confirm failure.
- [ ] Implement CrispEdit method branches, output naming, and cleanup behavior without affecting LoRA adapter branches.
- [ ] Run the focused loader tests and a dry-run of `scripts/test.sh --help`.

### Task 6: End-to-end verification and reproducibility checklist

**Files:**
- Modify: `run-lora/README.md`
- Create: `run-lora/docs/crispedit-reproduction.md`

- [ ] Document exact FineFineWeb generation command, calibration file schema, model-family target-module defaults, chunk/offload knobs, cosine scheduler requirement, and artifact paths.
- [ ] Run Python syntax compilation for changed modules.
- [ ] Run all focused CrispEdit tests.
- [ ] Run a reduced single-device training command with a tiny dataset/model or smallest available repository fixture; verify fixed-cache collection occurs once, training completes, and `<output_dir>/adapter_model` reloads.
- [ ] Run the existing task and world-knowledge evaluation entry points against the reduced artifact and record expected output files (`model_response.jsonl`, scores/perf JSON, and completion marker).
- [ ] Review the diff for accidental changes to unrelated dirty files.
