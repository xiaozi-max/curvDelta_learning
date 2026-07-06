import os
import torch
import gc
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

from peft import RandLoraConfig


ROOT_DIR = os.environ.get("ROOT_DIR")
if ROOT_DIR is None:
    raise RuntimeError("Please source common_utils/server_env.sh before running this script.")
MODEL_NAME = os.path.join(ROOT_DIR, "models", "Qwen", "Qwen3-0.6B-Base")

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def load_base_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cpu"
    )

def test_lora(rank):
    base_model = load_base_model()
    config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=TARGET_MODULES,
        lora_dropout=0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, config)
    trainable, total = count_params(model)
    del model, base_model
    gc.collect()
    return trainable, total

def test_randlora(rank):
    base_model = load_base_model()
    config = RandLoraConfig(
        r=rank,
        randlora_alpha=20 * rank, # as suggested officially
        target_modules=TARGET_MODULES,
        randlora_dropout=0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, config)
    trainable, total = count_params(model)
    del model, base_model
    gc.collect()
    return trainable, total


def main():
    print(f"Model: {MODEL_NAME}\n")

    # --- LoRA rank 128 ---
    print("=" * 70)
    print("LoRA rank=128")
    print("=" * 70)
    lora_trainable, lora_total = test_lora(rank=128)
    print(f"  Trainable: {lora_trainable:>12,}  ({100*lora_trainable/lora_total:.4f}%)")
    print()

    # --- RandLoRA with various ranks ---
    print("=" * 70)
    print("RandLoRA with various ranks")
    print("=" * 70)

    ranks_to_test = [1, 2, 3, 4]
    results = []
    for r in ranks_to_test:
        try:
            rand_trainable, rand_total = test_randlora(rank=r)
            ratio = rand_trainable / lora_trainable
            results.append((r, rand_trainable, rand_total, ratio))
            print(f"  rank={r:<4d}  Trainable: {rand_trainable:>12,}  "
                  f"({100*rand_trainable/rand_total:.4f}%)  "
                  f"ratio_vs_LoRA128: {ratio:.3f}x")
        except Exception as e:
            print(f"  rank={r:<4d}  FAILED: {e}")

    # --- Find closest match ---
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"LoRA rank=128 trainable params: {lora_trainable:,}")
    print()

    if results:
        closest = min(results, key=lambda x: abs(x[3] - 1.0))
        print(f"Closest RandLoRA to LoRA rank=128: rank={closest[0]}")
        print(f"  RandLoRA trainable: {closest[1]:,}")
        print(f"  LoRA trainable:     {lora_trainable:,}")
        print(f"  Ratio: {closest[3]:.3f}x")

if __name__ == "__main__":
    main()


'''
Model: ${ROOT_DIR}/models/Qwen/Qwen3-0.6B-Base

======================================================================
LoRA rank=128
======================================================================
  Trainable:   80,740,352  (11.9299%)

======================================================================
RandLoRA with various ranks
======================================================================
  rank=1     Trainable:  205,721,600  (25.6584%)  ratio_vs_LoRA128: 2.548x
  rank=2     Trainable:  102,961,152  (14.7295%)  ratio_vs_LoRA128: 1.275x
  rank=3     Trainable:   68,841,864  (10.3538%)  ratio_vs_LoRA128: 0.853x
  rank=4     Trainable:   51,580,928  (7.9646%)  ratio_vs_LoRA128: 0.639x

======================================================================
Summary
======================================================================
LoRA rank=128 trainable params: 80,740,352

Closest RandLoRA to LoRA rank=128: rank=3
  RandLoRA trainable: 68,841,864
  LoRA trainable:     80,740,352
  Ratio: 0.853x

--> In our paper Figure 1, we use rank=2 for RandLoRA instead to ensure vanilla LoRA does not exhibit higher number of trainable parameters
'''