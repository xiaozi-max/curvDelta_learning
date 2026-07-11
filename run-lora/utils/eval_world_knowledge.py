"""Evaluate world knowledge tasks with lm-eval after merging a LoRA adapter."""

import argparse
import json
import os
import shutil
import tempfile
from numbers import Number


TASKS = ["triviaqa", "nq_open", "webqs"]
TASK_METRIC_PREFERENCES = {
    "triviaqa": ["exact_match", "f1", "acc"],
    "nq_open": ["exact_match", "f1", "acc"],
    "webqs": ["f1", "exact_match", "acc"],
}

def resolve_base_model(
    peft,
    output_home,
    model_abbr,
    rank,
    output_path,
    model,
    timestamp,
    task=None,
):
    """Return the base path used by scripts/test.sh for this PEFT variant."""
    if peft == "PiSSA":
        return os.path.realpath(f"{output_home}/output/PiSSA-{model_abbr}-r{rank}")
    if peft == "MiLoRA":
        return os.path.realpath(f"{output_home}/output/MiLoRA-{model_abbr}-r{rank}")
    if peft == "OLoRA":
        return os.path.realpath(f"{output_path}/olora_residual_model")
    if peft == "InitAB":
        return os.path.realpath(
            f"{output_home}/output/InitAB-{model_abbr}-r{rank}-{timestamp}"
        )
    if peft == "LoRA-GA":
        if not task:
            raise ValueError("--task is required for LoRA-GA base model resolution")
        return os.path.realpath(f"{output_home}/output/LoRA-GA-{model_abbr}-r{rank}-{task}")
    return model


def metric_prefix(metric_key):
    return metric_key.split(",", 1)[0]


def is_number(value):
    return isinstance(value, Number) and not isinstance(value, bool)


def select_task_metric(task_name, task_result):
    """Pick the canonical metric from an lm-eval task result dict."""
    preferences = TASK_METRIC_PREFERENCES.get(task_name, ["exact_match", "f1", "acc"])
    numeric_metrics = {
        key: value for key, value in task_result.items() if is_number(value)
    }

    for wanted_metric in preferences:
        for metric_key, value in numeric_metrics.items():
            if metric_prefix(metric_key) == wanted_metric:
                return metric_key, value

    return None, None


def extract_world_knowledge_scores(results, tasks):
    """Extract scalar scores and selected lm-eval metric keys for each task."""
    task_results = results.get("results", {})
    scores = {}
    selected_metrics = {}
    raw_task_metrics = {}

    for task in tasks:
        task_result = task_results.get(task, {})
        metric_key, score = select_task_metric(task, task_result)
        scores[task] = round(float(score), 4) if score is not None else None
        selected_metrics[task] = metric_key
        raw_task_metrics[task] = sorted(task_result.keys())

    details = {
        "selected_metrics": selected_metrics,
        "raw_task_metrics": raw_task_metrics,
    }
    return scores, details


def parse_args():
    parser = argparse.ArgumentParser(description="World knowledge eval via lm-eval + vLLM")
    parser.add_argument("--model", type=str, required=True, help="Original base model path")
    parser.add_argument("--model_abbr", type=str, required=True)
    parser.add_argument("--lora", type=str, required=True, help="Trained adapter path")
    parser.add_argument("--peft", type=str, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--task", type=str, default=None, help="Training task name")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--output_home", type=str, default="./")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--timestamp", type=str, default="")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of samples per task for quick tests; remove or set to full later.",
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default="auto",
        help="lm-eval batch size, default auto.",
    )
    parser.add_argument(
        "--temp_path",
        type=str,
        default=None,
        help="Optional folder for the temporary merged model.",
    )
    return parser.parse_args()


def make_temp_dir(output_path, temp_path=None):
    if temp_path is None:
        return tempfile.mkdtemp(prefix="wk_merged_", dir=output_path)

    if os.path.exists(temp_path):
        raise FileExistsError(f"Temporary path already exists: {temp_path}")
    os.makedirs(temp_path)
    return temp_path


def merge_adapter_to_temp(base_model_path, adapter_path, temp_path):
    """Merge adapter with the same PEFT config handling as math_code.py."""
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[WK] Load base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="cpu")

    print(f"[WK] Load LoRA adapter from {adapter_path}")
    peft_config = PeftConfig.from_pretrained(adapter_path)
    if getattr(peft_config, "init_lora_weights", None) == "olora":
        peft_config.init_lora_weights = False
        print("[WK] OLoRA adapter detected: skipping olora_init")
        model = PeftModel.from_pretrained(base_model, adapter_path, config=peft_config)
    else:
        model = PeftModel.from_pretrained(base_model, adapter_path)

    merged_model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    print(f"[WK] Save merged model to {temp_path}")
    merged_model.save_pretrained(temp_path)
    tokenizer.save_pretrained(temp_path)

    del base_model, model, merged_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_lm_eval(model_path, tasks, gpus, batch_size, limit):
    import lm_eval

    tensor_parallel_size = len([gpu for gpu in gpus.split(",") if gpu.strip()])
    model_args = (
        f"pretrained={model_path},"
        f"tensor_parallel_size={tensor_parallel_size},"
        "dtype=auto,"
        "gpu_memory_utilization=0.7,"
        "max_model_len=1024,"
        "max_num_seqs=8"
    )
    print(f"[WK] Running lm_eval tasks={tasks}, limit={limit}, batch_size={batch_size}")
    return lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=5,
        batch_size=batch_size,
        device="cuda:0",
        # limit=100,
        log_samples=True,
    )


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")


def main():
    args = parse_args()
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    base_model_path = resolve_base_model(
        peft=args.peft,
        output_home=args.output_home,
        model_abbr=args.model_abbr,
        rank=args.rank,
        output_path=args.output_path,
        model=args.model,
        timestamp=args.timestamp,
        task=args.task,
    )
    temp_path = make_temp_dir(args.output_path, args.temp_path)

    try:
        print(f"[WK] Base model: {base_model_path}")
        print(f"[WK] Adapter:    {args.lora}")
        print(f"[WK] GPUs:       {args.gpus}")

        merge_adapter_to_temp(base_model_path, args.lora, temp_path)
        results = run_lm_eval(temp_path, TASKS, args.gpus, args.batch_size, args.limit)
        world_knowledge, details = extract_world_knowledge_scores(results, TASKS)

        output_file = os.path.join(args.output_path, "world_knowledge.json")
        details_file = os.path.join(args.output_path, "world_knowledge_lm_eval.json")

        write_json(output_file, world_knowledge)
        write_json(
            details_file,
            {
                "scores": world_knowledge,
                "details": details,
                "lm_eval": results,
            },
        )

        for task in TASKS:
            metric = details["selected_metrics"].get(task)
            print(f"[WK] {task}: {world_knowledge[task]} ({metric})")
        print(f"[WK] Saved scores: {output_file}")
        print(f"[WK] Saved raw lm-eval results: {details_file}")
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)
        print(f"[WK] Cleaned up {temp_path}")


if __name__ == "__main__":
    main()
