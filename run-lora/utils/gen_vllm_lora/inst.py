import argparse
import torch
import os
import json
import shutil

from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


def generate_prompt(instruction):
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n### Response:"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Path to base model or pre-merged model")
    parser.add_argument('--lora', type=str, default=None, help="Path to LoRA adapter; omit if model is already merged")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Directory containing IFEval/input_data.jsonl")
    parser.add_argument('--output_file', type=str, default="model_response.jsonl")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids, e.g. '0,1,2,3'")
    parser.add_argument('--temp_path', type=str, required=True, help="Folder to save temporary merged model")
    args = parser.parse_args()

    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    tensor_parallel_size = len(args.gpus.split(","))
    print(f"tensor_parallel_size: {tensor_parallel_size}")

    if args.lora is not None:
        print(f'Load base model from {args.model}')
        base_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cpu")
        print(f'Load LoRA from {args.lora}')
        peft_config = PeftConfig.from_pretrained(args.lora)
        if getattr(peft_config, 'init_lora_weights', None) == 'olora':
            peft_config.init_lora_weights = False
            print('OLoRA adapter detected: skipping olora_init to avoid double-residualization')
            model = PeftModel.from_pretrained(base_model, args.lora, config=peft_config)
        else:
            model = PeftModel.from_pretrained(base_model, args.lora)
        merged_model = model.merge_and_unload()
        print(merged_model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        print(f'Saving merged model to {args.temp_path}')
        merged_model.save_pretrained(args.temp_path)
        tokenizer.save_pretrained(args.temp_path)
        del base_model, merged_model
        torch.cuda.empty_cache()
        vllm_model_path = args.temp_path
        need_cleanup = True
    else:
        print(f'No LoRA specified, using pre-merged model at {args.model}')
        vllm_model_path = args.model
        need_cleanup = (args.model == args.temp_path)

    llm = LLM(model=vllm_model_path, tensor_parallel_size=tensor_parallel_size,
               gpu_memory_utilization=0.85)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    input_file = os.path.join(args.data_path, "IFEval", "input_data.jsonl")
    raw_instructions, inference_prompts = [], []
    print(f"Loading IFEval data from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            raw_instructions.append(data["prompt"])
            inference_prompts.append(generate_prompt(data["prompt"]))

    def batch_data(data_list, batch_size):
        return [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]

    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    for b_inference, b_raw in zip(
        batch_data(inference_prompts, args.batch_size),
        batch_data(raw_instructions, args.batch_size),
    ):
        completions = llm.generate(b_inference, sampling_params)
        with open(args.output_file, 'a', encoding='utf-8') as f:
            for raw_prompt, completion in zip(b_raw, completions):
                f.write(json.dumps({
                    "prompt": raw_prompt,
                    "response": completion.outputs[0].text,
                }, ensure_ascii=False) + "\n")

    if need_cleanup:
        print("Cleaning up temp merged model...")
        if os.path.exists(args.temp_path):
            shutil.rmtree(args.temp_path)
        else:
            print(f"Warning: temp path {args.temp_path} already removed, skipping cleanup")


if __name__ == '__main__':
    main()