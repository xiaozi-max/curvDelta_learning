# CrispEdit Baseline Design

## Goal

Add CrispEdit to `run-lora` as a full-model baseline that directly updates all configured transformer projection weights, uses one fixed FineFineWeb K-FAC cache, AdamW, and the repository's cosine learning-rate schedule.

## Scope and artifact

CrispEdit is an independent `train.py`/script branch, not a PEFT adapter variant. The model is loaded normally, all non-target parameters remain frozen, and the target projection weights are updated in place. The final artifact is a complete Hugging Face model saved in the same full-model location used by FFT/CurvatureDense evaluation (`<output_dir>/adapter_model`), so existing generation and world-knowledge pipelines can consume it with a small loader dispatch.

The target module set is configurable but defaults to every standard transformer projection in this repository: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`. No CrispEdit trimmed-layer or selected-matrix restriction is carried over.

## Fixed K-FAC cache

Before SFT training starts, collect uncentered activation and output-gradient factors for every target linear module using the FineFineWeb JSONL context. Collection is token-masked exactly as the existing `KFACTracker` path: `A = E[x x^T]` and `B = E[g g^T]` over valid next-token positions. Modules are processed in chunks; factor accumulators and eigenspaces may be moved to CPU between chunks to reduce peak GPU memory. All target modules receive exactly one cache and the cache is reused for the full run.

For each module, eigendecompose `A` and `B`, form the Kronecker eigenvalue matrix `outer(eig(A), eig(B))`, derive the energy threshold, and retain a boolean low-curvature mask `M` in the eigenbasis. The implementation must preserve the original CrispEdit orientation and mask transpose:

```text
grad_proj = U_B @ ((U_B.T @ grad @ U_A) * M.T) @ U_A.T
```

Only two-dimensional target weight gradients are projected. There is no cache recomputation, cache merging, optimizer `reset_cache`, or momentum reprojection because the cache is fixed for the entire run.

## Training integration

Reuse `run-lora/train.py` for model loading, task dataset loading, prompt/response tokenization, label masking, collation, logging, and checkpoint-independent output conventions. Add a dedicated full-model trainer/optimizer path that applies the projection immediately before AdamW's normal update. Use `torch.optim.AdamW` and always pass `--lr_scheduler_type cosine`; CrispEdit must not expose or inherit the existing curvature scheduler override.

The SFT task dataset remains separate from the FineFineWeb calibration dataset. Calibration data is generated/reused through `run-lora/utils/save_finefineweb.py` and is never mixed into the task dataset unless an experiment explicitly enables the existing replay feature.

## Evaluation and reproducibility

Update the unified shell dispatch and loaders so `CrispEdit` has a stable method name, output naming, full-model loading behavior, and the same task/world-knowledge scoring as other methods. The implementation documentation must point reproduceres to the CrispEdit modules, `run-lora/train.py`, `run-lora/scripts/train.sh`, `run-lora/scripts/test.sh`, `run-lora/utils/eval_world_knowledge.py`, `run-lora/utils/save_finefineweb.py`, and the existing K-FAC implementation in `run-lora/utils/curvature_lora.py`.

## Method distinction

CurvatureLoRA projects low-rank adapter updates using separate input/output high-curvature subspaces. CurvatureDense does the analogous bilateral projection for a dense delta and folds it into a full-rank adapter artifact. CrispEdit instead updates base weights directly and masks individual Kronecker eigen-direction pairs, allowing a pairwise decision that bilateral subspace removal cannot express. The shared K-FAC ingredients do not make the methods equivalent.

## Validation

The implementation must include focused tests for (1) Kronecker mask orientation and projected-gradient values, (2) chunked factor collection equivalence on a small deterministic model, (3) frozen/non-frozen parameter selection, (4) AdamW plus cosine scheduler wiring, and (5) full-model save/load plus the existing test/world-knowledge loader paths. A reduced end-to-end run should verify that the output artifact reloads and that no adapter-only files are required.
