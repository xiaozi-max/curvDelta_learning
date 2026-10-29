import argparse
import os
import json
import re
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='../output', help='Directory containing performance results')
    parser.add_argument('--task',   type=str, default='metamath',  help='Target task name')
    parser.add_argument('--model',  type=str, default='qwen-3-0.6b', help='Target model name (used for save path)')
    return parser.parse_args()


def extract_hyperparams_from_folder_name(folder_name):
    """Extract batch size, learning rate, and trial number from a subfolder name."""
    hyperparam = {}
    trial_num = 1

    bs_match = re.search(r'bs(\d+)', folder_name)
    if bs_match:
        hyperparam['batch_size'] = int(bs_match.group(1))

    lr_match = re.search(r'lr([\d.]+(?:e[+-]?\d+)?)', folder_name)
    if lr_match:
        hyperparam['lr'] = float(lr_match.group(1))

    trial_match = re.search(r'trial(\d+)', folder_name)
    if trial_match:
        trial_num = int(trial_match.group(1))

    # Replay runs carry a "-replay" suffix on the subfolder; tag them so they
    # aggregate as separate records instead of merging into the plain run.
    if "replay" in folder_name:
        hyperparam["replay"] = True

    return hyperparam, trial_num


def extract_info_from_exp_folder(folder_name, task, model):
    """
    Extract method, rank, epoch, and task from an experiment folder name.
    Expected format: {task}[-ep{n}]-{Method}-{model}-r{rank}
    Examples:
        metamath-LoRA-qwen-3-0.6b-r128        (epoch defaults to 1)
        metamath-ep3-LoRA-GA-qwen-3-0.6b-r128 (epoch=3, method=LoRA-GA)
    """
    rank_match = re.search(r'-r(\d+)$', folder_name)
    rank = int(rank_match.group(1)) if rank_match else None

    # Allow method names that contain hyphens (e.g. LoRA-GA)
    method_match = re.search(r'-([A-Z][A-Za-z0-9-]*)-' + re.escape(model) + r'-r\d+', folder_name)
    method = method_match.group(1).lower() if method_match else None

    if method_match:
        task_prefix = folder_name.split(f'-{method_match.group(1)}-', 1)[0]
        ep_match = re.search(r'-ep(\d+)$', task_prefix)
        epoch = int(ep_match.group(1)) if ep_match else 1
        task_name = re.sub(r'-ep\d+$', '', task_prefix)
    else:
        epoch = 1
        task_name = task

    return method, rank, task_name, epoch


def load_existing_data(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_data(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    args = parse_arguments()
    output_dir = Path(args.output)
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        return

    # Auto-generate save path: ./json/{model}/{task}/results.json
    save_dir  = os.path.join('./json', args.model.lower(), args.task)
    json_path = os.path.join(save_dir, 'results.json')

    data = load_existing_data(json_path)
    if args.task not in data:
        data[args.task] = {}

    for exp_folder in output_dir.iterdir():
        if not exp_folder.is_dir():
            continue

        method, rank, task_name, epoch = extract_info_from_exp_folder(exp_folder.name, args.task, args.model)

        if task_name != args.task:
            continue
        if method is None or rank is None:
            print(f"Skipping unrecognised folder: {exp_folder.name}")
            continue

        print(f"Processing: {exp_folder.name} (method={method}, rank={rank})")

        for sub_folder in exp_folder.iterdir():
            if not sub_folder.is_dir():
                continue

            hyperparam, trial_num = extract_hyperparams_from_folder_name(sub_folder.name)
            if not hyperparam:
                continue

            # Read world_knowledge.json independently (not reliant on perf.json)
            wk_data = None
            wk_file = sub_folder / 'world_knowledge.json'
            if wk_file.exists():
                with open(wk_file, 'r', encoding='utf-8') as f:
                    wk_data = json.load(f)
                print(f"  Found world_knowledge.json in {sub_folder.name}")

            # perf.json is the main source; if absent, fall back to WK-only record
            perf_file = sub_folder / 'perf.json'
            if not perf_file.exists():
                if wk_data is not None:
                    perf_data = {"world_knowledge": wk_data}
                else:
                    continue
            else:
                with open(perf_file, 'r', encoding='utf-8') as f:
                    perf_data = json.load(f)
                if wk_data is not None:
                    perf_data['world_knowledge'] = wk_data

            try:
                existing_list = data[args.task].setdefault(method, [])
                new_hyparam = {**hyperparam, 'rank': rank, 'epoch': epoch}
                acc_key = f'acc-{trial_num}'

                # Try to find an existing record with the same hyperparams
                existing_record = next(
                    (r for r in existing_list if r.get('hyparam') == new_hyparam and acc_key in r),
                    None
                )

                if existing_record is not None:
                    # Merge new fields into the existing record (incremental update)
                    old_data = existing_record[acc_key]
                    for new_key, new_val in perf_data.items():
                        if new_key not in old_data or old_data[new_key] is None:
                            old_data[new_key] = new_val
                    print(f"Updated: {sub_folder.name}")
                else:
                    record = {
                        'hyparam': new_hyparam,
                        acc_key: perf_data,
                    }
                    existing_list.append(record)
                    print(f"Added: {sub_folder.name}")

            except Exception as e:
                print(f"Error reading {perf_file}: {e}")

    save_data(json_path, data)
    print(f"Saved: {json_path}")


if __name__ == '__main__':
    main()