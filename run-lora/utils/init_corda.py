import argparse
import logging
import os

import torch
import transformers
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.config import CordaConfig
from peft.tuners.lora.corda import preprocess_corda

from corda_utils import load_corda_texts, make_corda_run_model, normalize_corda_method

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="CorDA initialization: save residual model and init adapter.")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--sub_task", type=str, required=True)
    parser.add_argument("--corda_method", type=str, default="kpm", choices=["kpm", "ipm", "Corda-KPM", "Corda-IPM"])
    parser.add_argument("--corda_num_samples", type=int, default=256)
    parser.add_argument("--corda_cache_dir", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=float, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--bits", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    method = normalize_corda_method(args.corda_method)

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.bits]

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(args.device)
    model.enable_input_require_grads()

    texts = load_corda_texts(
        data_path=args.data_path,
        task=args.sub_task,
        method=method,
        num_samples=args.corda_num_samples,
        dataset_split=args.dataset_split,
    )

    corda_kwargs = {"corda_method": method, "verbose": True}
    if args.corda_cache_dir:
        os.makedirs(args.corda_cache_dir, exist_ok=True)
        safe_task = args.sub_task.replace("/", "_").replace(":", "_")
        prefix = f"{safe_task}-{method}-r{args.lora_r}"
        corda_kwargs["cache_file"] = os.path.join(args.corda_cache_dir, f"{prefix}-svd.pt")

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="corda",
        corda_config=CordaConfig(**corda_kwargs),
    )

    logger.info("Running CorDA %s preprocessing on %d local calibration samples.", method, len(texts))
    run_model = make_corda_run_model(model, tokenizer, texts, args.model_max_length)
    preprocess_corda(model, lora_config, run_model=run_model)

    model = get_peft_model(model, lora_config)
    model.peft_config["default"].init_lora_weights = True

    init_adapter_dir = os.path.join(args.output_dir, "corda_init")
    logger.info("Saving CorDA init adapter to %s", init_adapter_dir)
    model.save_pretrained(init_adapter_dir, safe_serialization=False)

    residual = model.unload()
    logger.info("Saving CorDA residual model to %s", args.output_dir)
    residual.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("CorDA initialization complete.")


if __name__ == "__main__":
    main()
