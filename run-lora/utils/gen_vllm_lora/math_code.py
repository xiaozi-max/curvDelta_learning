import argparse
import re
import torch
import os
import json
import shutil

from peft import PeftModel, PeftConfig
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Path to base model or pre-merged model")
    parser.add_argument('--lora', type=str, default=None, help="Path to LoRA adapter; omit if model is already merged")
    parser.add_argument("--data_path", type=str, default="pissa-dataset", help="HuggingFace dataset path or local dir")
    parser.add_argument('--sub_task', nargs='+', help='List of sub-tasks (data_dir names) to load')
    parser.add_argument('--dataset_split', type=str, default="test")
    parser.add_argument('--output_file', type=str, default="model_response.jsonl")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids, e.g. '0,1,2,3'")
    parser.add_argument('--temp_path', type=str, required=True, help="Folder to save temporary merged model")
    args = parser.parse_args()

    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    tensor_parallel_size = len(args.gpus.split(","))
    print(f"tensor_parallel_size: {tensor_parallel_size}")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    if args.lora is not None:
        print(f'Load base model from {args.model}')
        base_model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cpu")
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
        vllm_model_path = args.temp_path
        need_cleanup = True
    else:
        print(f'No LoRA specified, using pre-merged model at {args.model}')
        vllm_model_path = args.model
        need_cleanup = (args.model == args.temp_path)

    llm = LLM(model=vllm_model_path, tensor_parallel_size=tensor_parallel_size,
               gpu_memory_utilization=0.7)

    def batch_data(data_list, batch_size=1):
        n = len(data_list) // batch_size
        batches = [data_list[i * batch_size:(i + 1) * batch_size] for i in range(n)]
        if len(data_list) % batch_size:
            batches.append(data_list[n * batch_size:])
        return batches

    if args.sub_task is None:
        dataset = load_dataset(args.data_path, split=args.dataset_split)
    else:
        all_test = []
        for task in args.sub_task:
            task_base = task.split(":", 1)[0]
            task_name = re.sub(r"-ep\d+$", "", task_base)
            print(task_name)
            ds = load_dataset(args.data_path, data_dir=task_name, split=args.dataset_split)
            all_test.append(ds)
        dataset = concatenate_datasets(all_test)

    batch_queries = batch_data(dataset["instruction"], batch_size=args.batch_size)
    batch_answers = batch_data(dataset["output"], batch_size=args.batch_size)
    batch_tasks   = batch_data(dataset["type"], batch_size=args.batch_size)

    for batch_query, batch_answer, batch_task in zip(batch_queries, batch_answers, batch_tasks):
        with torch.no_grad():
            completions = llm.generate(batch_query, sampling_params)
        with open(args.output_file, 'a', encoding='utf-8') as f:
            for query, completion, answer, task in zip(batch_query, completions, batch_answer, batch_task):
                f.write(json.dumps({
                    'type': task,
                    'query': query,
                    'output': completion.outputs[0].text,
                    'answer': answer,
                }, ensure_ascii=False) + "\n")

    if need_cleanup:
        print("Cleaning up temp merged model...")
        if os.path.exists(args.temp_path):
            shutil.rmtree(args.temp_path)
        else:
            print(f"Warning: temp path {args.temp_path} already removed, skipping cleanup")


if __name__ == '__main__':
    main()