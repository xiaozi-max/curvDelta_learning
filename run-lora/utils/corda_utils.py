import json
import logging
import re
from pathlib import Path
from typing import Callable, Iterable, List, Optional

logger = logging.getLogger(__name__)


def normalize_corda_method(peft_name: str, fallback: str = "kpm") -> str:
    peft_name = str(peft_name).lower()
    if peft_name.endswith("-kpm") or peft_name == "kpm":
        return "kpm"
    if peft_name.endswith("-ipm") or peft_name == "ipm":
        return "ipm"
    fallback = str(fallback).lower()
    if fallback not in {"kpm", "ipm"}:
        raise ValueError(f"Unsupported Corda method: {fallback}")
    return fallback


def clean_task_name(task: str) -> str:
    task = str(task)
    task = re.sub(r"-ep[0-9]+$", "", task)
    task = task.split(":", 1)[0]
    return task


def _iter_json_records(path: Path) -> Iterable[dict]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL record at {path}:{line_no}: {exc}") from exc
                if isinstance(record, dict):
                    yield record
                else:
                    raise ValueError(f"Expected JSON object at {path}:{line_no}, got {type(record).__name__}.")
        return

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                for record in value:
                    if isinstance(record, dict):
                        yield record
                return
        yield data
    elif isinstance(data, list):
        for record in data:
            if isinstance(record, dict):
                yield record
            else:
                raise ValueError(f"Expected JSON object in {path}, got {type(record).__name__}.")
    else:
        raise ValueError(f"Expected JSON object or list in {path}, got {type(data).__name__}.")


def _record_to_text(record: dict) -> Optional[str]:
    text = record.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    instruction = record.get("instruction")
    output = record.get("output")
    if isinstance(instruction, str) and instruction.strip():
        if isinstance(output, str) and output.strip():
            return f"{instruction.strip()}\n{output.strip()}"
        return instruction.strip()

    question = record.get("question")
    answer = record.get("answer")
    if isinstance(question, str) and question.strip():
        if isinstance(answer, str) and answer.strip():
            return f"Question: {question.strip()}\nAnswer: {answer.strip()}"
        return f"Question: {question.strip()}\nAnswer:"

    input_text = record.get("input")
    response = record.get("response")
    if isinstance(input_text, str) and input_text.strip():
        if isinstance(response, str) and response.strip():
            return f"{input_text.strip()}\n{response.strip()}"
        return input_text.strip()

    return None


def _candidate_calibration_files(data_path: Path, task_name: str, method: str, dataset_split: str) -> List[Path]:
    if data_path.is_file():
        return [data_path]

    names = [
        f"corda_{method}.jsonl",
        f"corda-{method}.jsonl",
        "corda.jsonl",
        f"{method}.jsonl",
        f"{dataset_split}.jsonl",
        f"{dataset_split}.json",
        "train.jsonl",
        "train.json",
        "test.jsonl",
        "test.json",
        "finefineweb_validation.jsonl",
    ]
    roots = [
        data_path / "corda" / method / task_name,
        data_path / "corda" / task_name,
        data_path / "corda" / method,
        data_path / task_name,
        data_path,
    ]
    candidates = []
    for root in roots:
        for name in names:
            candidates.append(root / name)
    return candidates


def load_corda_texts(
    data_path: str,
    task: str,
    method: str,
    num_samples: int,
    dataset_split: str = "train",
) -> List[str]:
    task_name = clean_task_name(task)
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Corda calibration path not found: {path}")

    selected_file = None
    for candidate in _candidate_calibration_files(path, task_name, method, dataset_split):
        if candidate.exists() and candidate.is_file():
            selected_file = candidate
            break

    if selected_file is None:
        searched = "\n".join(str(p) for p in _candidate_calibration_files(path, task_name, method, dataset_split)[:10])
        raise FileNotFoundError(f"No local Corda calibration file found under {path}. First searched paths:\n{searched}")

    texts = []
    for record in _iter_json_records(selected_file):
        text = _record_to_text(record)
        if text:
            texts.append(text)
        if len(texts) >= int(num_samples):
            break
    if not texts:
        raise ValueError(f"Corda calibration file contains no usable text: {selected_file}")
    logger.info("Loaded %d Corda calibration texts from %s", len(texts), selected_file)
    return texts


def make_corda_run_model(
    model,
    tokenizer,
    texts: List[str],
    max_length: int,
) -> Callable[[], None]:
    import torch

    def run_model() -> None:
        model.eval()
        device = getattr(model, "device", None)
        if device is None:
            device = next(model.parameters()).device
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                model(**inputs)

    return run_model
