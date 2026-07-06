# Repository Guidelines

## Project Structure & Module Organization

This repository supports the paper "Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning." Main entry points:

- `run-lora/`: training, generation, scoring, and result aggregation for LoRA variants.
- `run-lora/scripts/`: runnable experiment scripts by model family, plus shared `train.sh` and `test.sh`.
- `run-lora/utils/`: initialization, generation, evaluation, and task-specific processing utilities.
- `run-lora/configs/`: DeepSpeed ZeRO configuration files.
- `run-hessian/`: Lanczos-based Hessian eigenvalue estimation.
- `run-hessian/eigen_results/`: checked-in example Hessian outputs.
- `assets/` and `practical-heuristics/assets/`: figures used by README documentation.
- `awesome-lora-paper-list/`: LoRA bibliography summary and plotting script.

Keep large generated artifacts, including datasets, checkpoints, adapters, and experiment outputs, out of git unless intentionally curated.

## Build, Test, and Development Commands

Run a representative LoRA tuning experiment:

```bash
cd run-lora
bash ./scripts/qwen/math.sh
```

Aggregate LoRA results:

```bash
cd run-lora/exp_record
python3 get_record.py --model qwen-3-0.6b --task metamath --output ../output
python3 summary.py --record_path ./json/qwen-3-0.6b/metamath/results.json
```

Run Hessian estimation:

```bash
cd run-hessian
bash scripts/run.sh --model Qwen/Qwen3-0.6B-Base --peft lora --rank 128
```

## Coding Style & Naming Conventions

Use Python 3.12, 4-space indentation, and snake_case for functions, files, and variables. Keep flags consistent with names such as `--data_path`, `--dataset_field`, `--lora_r`, and `--cuda_devices`. Prefer utilities under `utils/` over duplicating experiment logic in shell scripts. Preserve dataclass-based argument definitions in `train.py` when adding training options.

## Testing Guidelines

There is no standalone unit-test suite. Validate changes with the smallest relevant workflow: a targeted training script, `scripts/test.sh`, `exp_record` aggregation, or a reduced Hessian command with small `--b` and `--max_iter`. For evaluation changes, check `perf.json`, `model_response.jsonl`, and summary output. For Hessian changes, confirm JSON files under `run-hessian/eigen_results/` contain valid eigenvalue histories.

## Commit & Pull Request Guidelines

Recent commits use short, imperative, lowercase messages such as `update readme` and `correct data path`. Keep commits focused and mention affected areas when useful, for example `fix qwen math data path`. Pull requests should include purpose, changed commands or configs, environment differences, output locations, and reproducibility caveats. Include figures only when documentation or plotted assets change.

## Security & Configuration Tips

Do not commit Hugging Face tokens, W&B keys, local storage paths, downloaded datasets, or model weights. Document any hard-coded machine paths in the relevant README or replace them with flags before submitting.
