import argparse
import json
import numpy as np
import os
import re
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description='Display accuracy vs learning rate statistics from JSON experiment records')
    parser.add_argument('--record_path', type=str, required=True,
                        help='Path to the JSON experiment record file')
    parser.add_argument('--min_lr', type=float, default=None,
                        help='Minimum learning rate to include')
    parser.add_argument('--max_lr', type=float, default=None,
                        help='Maximum learning rate to include')
    parser.add_argument('--default_lr_only', type=bool, default=True,
                        help='Only include LRs with mantissa in [1.1247, 2.0, 3.5566, 6.3246]')
    return parser.parse_args()


def get_mantissa(lr):
    if lr == 0:
        return 0
    exponent = int(np.floor(np.log10(abs(lr))))
    return lr / (10 ** exponent)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _fmt_pct_list(vals):
    return "[" + ", ".join(f"{v * 100:.2f}%" for v in vals) + "]"


def load_data(json_path, task, min_lr=None, max_lr=None, default_lr_only=True):
    """
    Load JSON and build per-method series and subtask breakdown.
    Returns:
        series: {label: {lrs, means, stds, runs}}
        breakdown_data: {label: {lr: {subtask: [vals]}}}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    default_mantissas = [1.1247, 2.0, 3.5566, 6.3246]

    COMMONSENSE_SUBTASKS = [
        "boolq", "piqa", "social_i_qa", "hellaswag",
        "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa",
    ]

    label_map = {
        'lora':    'LoRA',
        'pissa':   'PiSSA',
        'milora':  'MiLoRA',
        'dora':    'DoRA',
        'initab':  'InitAB',
        'lora-ga': 'LoRA-GA',
        'oplora':  'OPLoRA',
    }

    series = {}
    breakdown_data = {}

    for top_key, methods in data.items():
        for method_name, records in methods.items():
            label = label_map.get(method_name.lower(), method_name)

            if label not in breakdown_data:
                breakdown_data[label] = {}

            lr_acc_dict = {}

            for rec in records:
                lr = rec.get('hyparam', {}).get('lr')
                if lr is None:
                    continue

                if default_lr_only:
                    mantissa = get_mantissa(lr)
                    if not any(np.isclose(mantissa, m, rtol=1e-4) for m in default_mantissas):
                        continue
                else:
                    if min_lr is not None and lr < min_lr:
                        continue
                    if max_lr is not None and lr > max_lr:
                        continue

                if lr not in breakdown_data[label]:
                    breakdown_data[label][lr] = {}

                for k, v in rec.items():
                    if not (k.startswith('acc-') and isinstance(v, dict)):
                        continue
                    trial_num = int(k.split('-')[1])

                    if 'metamath' in task:
                        math_acc   = v.get('math')
                        gsm8k_acc  = v.get('gsm8k')
                        if math_acc is None or gsm8k_acc is None:
                            continue
                        acc = (math_acc + gsm8k_acc) / 2.0
                        breakdown_data[label][lr].setdefault('MATH',  []).append(math_acc)
                        breakdown_data[label][lr].setdefault('GSM8K', []).append(gsm8k_acc)

                    elif 'python' in task:
                        humaneval_acc = v.get('humaneval', [None])[0]
                        mbpp_acc      = v.get('mbpp',      [None])[0]
                        if humaneval_acc is None or mbpp_acc is None:
                            continue
                        acc = (humaneval_acc + mbpp_acc) / 2.0
                        breakdown_data[label][lr].setdefault('HumanEval', []).append(humaneval_acc)
                        breakdown_data[label][lr].setdefault('MBPP',      []).append(mbpp_acc)

                    elif 'instruction' in task:
                        strict_prompt = v.get('ifeval_strict_prompt')
                        strict_inst   = v.get('ifeval_strict_inst')
                        if strict_prompt is None or strict_inst is None:
                            continue
                        acc = (strict_prompt + strict_inst) / 2.0
                        breakdown_data[label][lr].setdefault('IFEval-Strict-Prompt', []).append(strict_prompt)
                        breakdown_data[label][lr].setdefault('IFEval-Strict-Inst',   []).append(strict_inst)

                    elif task == 'commonsense':
                        accs = [v.get(t) for t in COMMONSENSE_SUBTASKS]
                        if any(a is None for a in accs):
                            continue
                        acc = sum(accs) / len(accs)

                    else:
                        acc = 0.0

                    # World knowledge subtasks (always extracted when present)
                    wk_data = v.get('world_knowledge')
                    if isinstance(wk_data, dict):
                        wk_vals = []
                        for wk_task, wk_score in wk_data.items():
                            if wk_score is not None:
                                breakdown_data[label][lr].setdefault(wk_task, []).append(wk_score)
                                wk_vals.append(wk_score)
                        if wk_vals:
                            breakdown_data[label][lr].setdefault('WK-Avg', []).append(
                                sum(wk_vals) / len(wk_vals)
                            )

                    lr_acc_dict.setdefault(lr, []).append((trial_num, acc))

            if not lr_acc_dict:
                continue

            lrs, means, stds = [], [], []
            for lr, pairs in lr_acc_dict.items():
                pairs.sort(key=lambda x: x[0])
                vals = [a for _, a in pairs]
                arr = np.array(vals)
                lrs.append(lr)
                means.append(arr.mean())
                stds.append(arr.std(ddof=1) if len(vals) > 1 else 0.0)

            lrs   = np.array(lrs)
            means = np.array(means)
            stds  = np.array(stds)
            order = np.argsort(lrs)

            runs_sorted = {}
            for i in order:
                lr = float(lrs[i])
                pairs = lr_acc_dict[lrs[i]]
                pairs.sort(key=lambda x: x[0])
                runs_sorted[lr] = [a for _, a in pairs]

            series[label] = {
                'lrs':   lrs[order],
                'means': means[order],
                'stds':  stds[order],
                'runs':  runs_sorted,
            }

    return series, breakdown_data


def normalize_lr_bounds(min_lr, max_lr):
    if min_lr is not None and max_lr is not None and min_lr > max_lr:
        min_lr, max_lr = max_lr, min_lr
    return min_lr, max_lr


def extract_config_from_path(record_path):
    """Extract task and model from path: .../json/{model}/{task}/results.json"""
    parts = Path(record_path).parts
    try:
        json_idx = next(i for i, p in enumerate(parts) if p == 'json')
        model = parts[json_idx + 1]
        task  = parts[json_idx + 2]
        return task, model
    except (StopIteration, IndexError):
        return None, None


def print_subtask_breakdown(breakdown_data):
    if not breakdown_data:
        return

    first_method = next(iter(breakdown_data))
    if not breakdown_data[first_method]:
        return
    first_lr = next(iter(breakdown_data[first_method]))
    subtask_keys = sorted(breakdown_data[first_method][first_lr].keys())

    if not subtask_keys:
        return

    for st in subtask_keys:
        print("\n" + "=" * 70)
        print(f"SUB-TASK PERFORMANCE: {st}")
        print("=" * 70)

        for method, lr_data in breakdown_data.items():
            print(f"\n[{method}]")
            print(f"{'LR':>12} | {'Mean':>10} | {'StdDev':>10} | Runs")
            print("-" * 70)

            for lr in sorted(lr_data.keys()):
                vals = lr_data[lr].get(st, [])
                if vals:
                    arr  = np.array(vals)
                    mean = arr.mean()
                    std  = arr.std(ddof=1) if len(vals) > 1 else 0.0
                    print(f"{lr:>12.2e} | {_fmt_pct(mean):>10} | {_fmt_pct(std):>10} | {_fmt_pct_list(vals)}")
                else:
                    print(f"{lr:>12.2e} | {'N/A':>10} | {'N/A':>10} | []")


def print_per_lr_table(series: dict):
    print("\n" + "=" * 70)
    print("AGGREGATED PERFORMANCE (per method)")
    print("=" * 70)

    for method, data in series.items():
        lrs   = data['lrs']
        means = data['means']
        stds  = data['stds']
        runs  = data.get('runs', {})
        order = np.argsort(lrs)

        print(f"\n[{method}]")
        print(f"{'LR':>12} | {'Mean':>10} | {'StdDev':>10} | Runs")
        print("-" * 70)
        for idx in order:
            lr       = float(lrs[idx])
            run_vals = runs.get(lr, [])
            print(f"{lr:>12.2e} | {_fmt_pct(means[idx]):>10} | {_fmt_pct(stds[idx]):>10} | {_fmt_pct_list(run_vals)}")


WK_TASKS = ['triviaqa', 'nq_open', 'webqs']


def print_world_knowledge_breakdown(breakdown_data):
    """Print world knowledge subtask breakdown if data exists."""
    has_wk = any(
        any(wk in lr_data for lr_data in method_data.values())
        for method_data in breakdown_data.values()
        for wk in WK_TASKS
    )
    if not has_wk:
        return

    all_wk_tasks = ['WK-Avg'] + WK_TASKS

    for st in all_wk_tasks:
        print("\n" + "=" * 70)
        print(f"WORLD KNOWLEDGE: {st}")
        print("=" * 70)

        for method, lr_data in breakdown_data.items():
            has_any = any(st in d for d in lr_data.values())
            if not has_any:
                continue
            print(f"\n[{method}]")
            print(f"{'LR':>12} | {'Mean':>10} | {'StdDev':>10} | Runs")
            print("-" * 70)
            for lr in sorted(lr_data.keys()):
                vals = lr_data[lr].get(st, [])
                if vals:
                    arr  = np.array(vals)
                    mean = arr.mean()
                    std  = arr.std(ddof=1) if len(vals) > 1 else 0.0
                    print(f"{lr:>12.2e} | {_fmt_pct(mean):>10} | {_fmt_pct(std):>10} | {_fmt_pct_list(vals)}")
                else:
                    print(f"{lr:>12.2e} | {'N/A':>10} | {'N/A':>10} | []")


def main():
    args = parse_arguments()
    args.min_lr, args.max_lr = normalize_lr_bounds(args.min_lr, args.max_lr)

    if not os.path.exists(args.record_path):
        print(f"Error: Record file '{args.record_path}' not found.")
        return

    task, model = extract_config_from_path(args.record_path)

    if task is None:
        print(f"Warning: Could not extract config from path '{args.record_path}'")
        task, model = "unknown", "unknown"

    series, breakdown_data = load_data(
        args.record_path, task, args.min_lr, args.max_lr, args.default_lr_only
    )

    if not series:
        print("No valid data series found.")
        return

    print(f"\nAnalyzing: {args.record_path}")
    print(f"Task: {task}, Model: {model}")
    print(f"Data series found: {list(series.keys())}")

    if task in ('metamath', 'python', 'instruction'):
        print_subtask_breakdown(breakdown_data)

    print_world_knowledge_breakdown(breakdown_data)

    print_per_lr_table(series)

    print("\n" + "=" * 70)
    print("BEST PERFORMANCE FOR EACH METHOD")
    print("=" * 70)

    best_results = []
    for method, data in series.items():
        best_idx      = np.argmax(data['means'])
        best_accuracy = data['means'][best_idx] * 100
        best_lr       = data['lrs'][best_idx]
        best_std      = data['stds'][best_idx] * 100
        num_runs      = len(data['runs'].get(float(best_lr), []))

        best_results.append({
            'method':   method,
            'accuracy': best_accuracy,
            'lr':       best_lr,
            'std':      best_std,
            'num_runs': num_runs,
        })
        print(f"{method:12s} | Best: {best_accuracy:.2f}% (±{best_std:.2f}%) | "
              f"LR: {best_lr:.2e} | Runs: {num_runs}")

    print("\n" + "=" * 70)
    print("RANKING (by Best Accuracy)")
    print("=" * 70)

    for i, result in enumerate(sorted(best_results, key=lambda x: x['accuracy'], reverse=True), 1):
        print(f"{i}. {result['method']:12s} | {result['accuracy']:.2f}% (±{result['std']:.2f}%) | "
              f"LR: {result['lr']:.2e} | Runs: {result['num_runs']}")

    print("=" * 70)


if __name__ == '__main__':
    main()