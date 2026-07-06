# Train various LoRA methods with learning rate tuning

## README Table of Contents
- [Environment Setup](#environment-setup)
  - [1. Main Environment](#1-main-environment)
  - [2. LoRA-GA Environment](#2-lora-ga-environment)
  - [3. GraLoRA Environment](#3-gralora-environment)
- [Code Strucutre](#code-structure)
- [Download Fine-tuning Datasets](#download-fine-tuning-datasets)
  - [1. Mathematical Reasoning and Code Generation](#1-mathematical-reasoning-and-code-generation)
  - [2. Commonsense Reasoning](#2-commonsense-reasoning)
  - [3. Instruction Following](#3-instruction-following)
- [Run Learning Rate Tuning](#run-learning-rate-tuning)
- [On Reproducibility](#on-reproducibility)
- [Known Issues](#known-issues)

## Environment Setup

> [!NOTE]
> **Since different methods require different package versions, we maintain separate conda environments to avoid dependency conflicts.**
>
> - You are welcome to find a way to set up an environment that can run all methods compatibly (e.g., by upgrading to the latest PEFT version and modifying the code accordingly).
> - However, results under different PEFT versions may differ slightly, as described in [On Reproducibility](#on-reproducibility).


### *1. Main Environment*
For most methods (LoRA, OLoRA, PiSSA, MiLoRA, Init[AB], DoRA, LoFT):

```bash
# Create and activate conda environment
conda create -n lora-env python=3.12 -y
conda activate lora-env

# Install PyTorch
pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# Install Flash-Attention
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip3 install flash_attn-2.8.2+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# Install requirements
pip3 install -r requirements.txt
```

> ***Adjust the PyTorch installation command based on your CUDA version (see [here](https://pytorch.org/get-started/previous-versions/)).***  
> ***For Flash Attention, adjust to download the wheel that matches your specific cuda version [here](https://github.com/Dao-AILab/flash-attention/releases) (see [here](https://til.simonwillison.net/python/installing-flash-attention) for more instruction).***

### *2. LoRA-GA Environment*
LoRA-GA requires a custom-modified version of PEFT (see [LoRA-GA's codebase](https://github.com/Outsider565/LoRA-GA)).

Download [LoRA-GA's custom PEFT](https://github.com/Outsider565/LoRA-GA/tree/main/peft):
```bash
svn export https://github.com/Outsider565/LoRA-GA/trunk/peft
```

Clone the main environment and install the custom PEFT:
```bash
conda create --name lora-env-loraga --clone lora-env
conda activate lora-env-loraga
pip uninstall peft -y
pip install -e ./peft
```

### *3. GraLoRA Environment*
GraLoRA requires a newer version of PEFT:

```bash
conda create --name lora-env-gralora --clone lora-env
conda activate lora-env-gralora
pip install -U git+https://github.com/huggingface/peft.git@main  # installed directly from source
```

## Code Structure
```
run-lora/
├── configs/
│   ├── ds_config_zero2_no_offload.json  # DeepSpeed ZeRO-2
│   └── ds_config_zero3.json             # DeepSpeed ZeRO-3
│
├── scripts/
│   ├── qwen/math.sh      # example: Qwen3-0.6B on MetaMath
│   ├── gemma/{math,code}.sh
│   ├── llama/{math,code}.sh
│   ├── train.sh          # unified training entry point → calls train.py via DeepSpeed
│   └── test.sh           # unified evaluation entry point → generation + scoring
│
├── utils/
│   ├── init_pissa.py     # PiSSA init
│   ├── init_milora.py    # MiLoRA init
│   ├── init_AB_util.py   # InitAB init
│   ├── curvature_lora.py             # CurvatureLoRA forward, KFAC collection, and folding
│   ├── save_finefineweb_curvature.py # FineFineWeb-validation curvature anchor preprocessing
│   ├── gen_vllm_lora.py              # vLLM generation for model testing
│   ├── test_acc.py                   # accuracy scoring (math / commonsense / IFEval)
│   └── code_process.py               # code scoring (HumanEval / MBPP via evalplus)
│   
├── exp_record/
│   ├── get_record.py     # scan output/ → ./json/{model}/{task}/results.json
│   └── summary.py        # read results.json → print structured summary in terminal
│
└── train.py              # HuggingFace Trainer + DeepSpeed, all PEFT methods
```

## Download Fine-tuning Datasets

All commands below assume you are running from the `run-lora` directory:

```bash
cd run-lora
```

You can download all datasets with:

```bash
bash ./scripts/download_datasets.sh
```

### *1. Mathematical Reasoning and Code Generation*

We use the preprocessed datasets released by the PiSSA authors (see the [HuggingFace page](https://huggingface.co/datasets/fxmeng/pissa-dataset)):

```bash
huggingface-cli download --repo-type dataset --resume-download fxmeng/pissa-dataset \
    --include "metamath/*" "python/*" \
    --local-dir ./lora-dataset/pissa-dataset
```

CurvatureLoRA also needs a raw-text curvature anchor set. The default path expected by `scripts/train.sh` is `./lora-dataset/curvature/finefineweb_validation_16384_seed42.jsonl`:

```bash
python utils/save_finefineweb_curvature.py
```

### *2. Commonsense Reasoning*

#### Training dataset

```bash
mkdir -p ./lora-dataset/llm-adapters-dataset
wget -O ./lora-dataset/llm-adapters-dataset/commonsense_15k.json \
    https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/main/ft-training_set/commonsense_15k.json
```

#### Testing dataset
```bash
for dataset in boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa; do
    mkdir -p ./lora-dataset/llm-adapters-dataset/${dataset}
    wget -O ./lora-dataset/llm-adapters-dataset/${dataset}/test.json \
        https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/main/dataset/${dataset}/test.json
done
```

### *3. Instruction Following*

#### Training dataset

```bash
mkdir -p ./lora-dataset/instruction-dataset
python3 -c '
from datasets import load_dataset
import json
d = load_dataset("yahma/alpaca-cleaned", split="train")
res = [{"instruction": i["instruction"], "input": i["input"], "output": i["output"]} for i in d]
json.dump(res, open("./lora-dataset/instruction-dataset/alpaca_clean.json", "w"), indent=2)
'
```

#### Testing dataset

+ Download the IFEval evaluation data from the [Google Research repository](https://github.com/google-research/google-research/tree/master/instruction_following_eval):

```bash
wget -O ./lora-dataset/instruction-dataset/input_data.jsonl \
    https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/data/input_data.jsonl
```

## Run Learning Rate Tuning

### *1. Qwen3-0.6B on MetaMath (i.e., results reported in paper Figure 1)*

```bash
bash ./scripts/qwen/math.sh 2>&1 | tee "./scripts/qwen/math-test-$(date +"%Y%m%d-%H%M%S").txt"
```

Please see comments in `./scripts/qwen/math.sh` for line-by-line explanation

> ***For RandLoRA, note that its trainable parameter count is inversely proportional to the LoRA rank parameter r, as illustrated [here](https://github.com/huggingface/peft/tree/eaecab993365da1c939347155a961bc00168d672/examples/randlora_finetuning).***
> ***We provide example code in `count_randlora.py` to find an r setting that matches LoRA.***

### *2. Aggregate Results*

+ By default, the following fine-tuning results will be saved to `./output`:
    + `adapter_model`: trained LoRA adapters; will be deleted by default after testing to free-up disk space
    + `perf.json`: model performance on testing dataset
    + `model_response.jsonl`: model response on testing dataset
    + `train_loss,json`: training loss log
+ Adjust `output_home` in `./scripts/qwen/math.sh` to change to other directories.
+ To aggregate all the results, please run through the following.

```bash
cd exp_record
```

#### *2.1. Collect all hyperparameter search results into a JSON file*
```bash
python3 get_record.py \
    --model qwen-3-0.6b \
    --task metamath \
    --output ../output
```

+ Results are saved to `./json/{model}/{task}/results.json` automatically.

#### *2.2 Show structured output in the terminal*
```bash
python3 summary.py --record_path ./json/qwen-3-0.6b/metamath/results.json
```

+ An example of our results:
```text
Task: metamath, Model: qwen-3-0.6b
Data series found: ['DoRA', 'LoRA', 'InitAB', 'PiSSA', 'MiLoRA']

======================================================================
SUB-TASK PERFORMANCE: GSM8K
======================================================================

[DoRA]
          LR |       Mean |     StdDev | Runs
----------------------------------------------------------------------
    2.00e-06 |     42.61% |      1.69% | [44.50%, 41.24%, 42.08%]
    3.56e-06 |     45.08% |      2.90% | [48.14%, 44.73%, 42.38%]
    6.32e-06 |     46.40% |      0.92% | [46.25%, 47.38%, 45.56%]
    1.12e-05 |     57.19% |      1.98% | [57.85%, 54.97%, 58.76%]
    2.00e-05 |     63.33% |      1.21% | [63.23%, 64.59%, 62.17%]
    3.56e-05 |     64.92% |      0.53% | [64.37%, 64.97%, 65.43%]
    6.32e-05 |     65.63% |      0.38% | [65.28%, 66.03%, 65.58%]
    1.12e-04 |     65.83% |      0.23% | [66.03%, 65.58%, 65.88%]
    2.00e-04 |     65.88% |      0.66% | [65.58%, 66.64%, 65.43%]
    3.56e-04 |     66.95% |      0.60% | [67.63%, 66.72%, 66.49%]
    6.32e-04 |     65.40% |      0.56% | [66.03%, 65.20%, 64.97%]
    1.12e-03 |     59.51% |      0.88% | [60.50%, 59.21%, 58.83%]
    2.00e-03 |     15.87% |     25.13% | [44.88%, 1.36%, 1.36%]
```
<details>

<summary>click for more...</summary>
   

    [LoRA]
            LR |       Mean |     StdDev | Runs
    ----------------------------------------------------------------------
        2.00e-06 |     17.21% |      0.72% | [17.13%, 17.97%, 16.53%]
        3.56e-06 |     21.84% |      0.93% | [21.15%, 21.46%, 22.90%]
        6.32e-06 |     32.62% |      1.14% | [31.54%, 32.52%, 33.81%]
        1.12e-05 |     53.53% |      2.19% | [55.88%, 51.55%, 53.15%]
        2.00e-05 |     63.91% |      0.20% | [64.14%, 63.76%, 63.84%]
        3.56e-05 |     64.67% |      0.27% | [64.44%, 64.97%, 64.59%]
        6.32e-05 |     65.60% |      0.27% | [65.58%, 65.88%, 65.35%]
        1.12e-04 |     65.50% |      0.49% | [66.03%, 65.05%, 65.43%]
        2.00e-04 |     66.24% |      0.50% | [66.49%, 66.57%, 65.66%]
        3.56e-04 |     66.03% |      0.13% | [66.11%, 65.88%, 66.11%]
        6.32e-04 |     65.96% |      0.42% | [66.41%, 65.88%, 65.58%]
        1.12e-03 |     60.00% |      1.03% | [58.91%, 60.12%, 60.96%]
        2.00e-03 |      1.34% |      0.46% | [1.29%, 1.82%, 0.91%]

    [InitAB]
            LR |       Mean |     StdDev | Runs
    ----------------------------------------------------------------------
        2.00e-06 |     36.29% |      4.33% | [39.20%, 31.31%, 38.36%]
        3.56e-06 |     40.26% |      3.19% | [40.26%, 37.07%, 43.44%]
        6.32e-06 |     50.09% |      3.48% | [51.71%, 46.10%, 52.46%]
        1.12e-05 |     61.76% |      1.89% | [61.03%, 60.35%, 63.91%]
        2.00e-05 |     65.20% |      0.15% | [65.20%, 65.05%, 65.35%]
        3.56e-05 |     65.43% |      0.69% | [65.58%, 66.03%, 64.67%]
        6.32e-05 |     65.48% |      0.22% | [65.73%, 65.35%, 65.35%]
        1.12e-04 |     66.42% |      0.20% | [66.49%, 66.57%, 66.19%]
        2.00e-04 |     66.09% |      0.65% | [66.57%, 65.35%, 66.34%]
        3.56e-04 |     66.99% |      0.80% | [67.85%, 66.87%, 66.26%]
        6.32e-04 |     64.29% |      0.61% | [63.68%, 64.90%, 64.29%]
        1.12e-03 |     59.39% |      0.59% | [60.05%, 58.91%, 59.21%]
        2.00e-03 |      0.83% |      0.99% | [0.23%, 1.97%, 0.30%]

    [PiSSA]
            LR |       Mean |     StdDev | Runs
    ----------------------------------------------------------------------
        2.00e-06 |     44.15% |      1.71% | [43.44%, 42.91%, 46.10%]
        3.56e-06 |     56.15% |      0.38% | [56.41%, 55.72%, 56.33%]
        6.32e-06 |     63.35% |      0.35% | [63.15%, 63.15%, 63.76%]
        1.12e-05 |     65.10% |      0.09% | [65.05%, 65.05%, 65.20%]
        2.00e-05 |     66.11% |      0.15% | [66.11%, 65.96%, 66.26%]
        3.56e-05 |     66.57% |      0.42% | [66.19%, 67.02%, 66.49%]
        6.32e-05 |     66.01% |      0.19% | [66.19%, 66.03%, 65.81%]
        1.12e-04 |     66.57% |      0.67% | [65.81%, 66.79%, 67.10%]
        2.00e-04 |     63.51% |      0.57% | [63.76%, 62.85%, 63.91%]
        3.56e-04 |     59.14% |      0.80% | [59.44%, 58.23%, 59.74%]
        6.32e-04 |     53.05% |      0.76% | [52.24%, 53.75%, 53.15%]
        1.12e-03 |     44.00% |      2.30% | [41.93%, 46.47%, 43.59%]
        2.00e-03 |     29.54% |      1.05% | [30.33%, 29.95%, 28.35%]

    [MiLoRA]
            LR |       Mean |     StdDev | Runs
    ----------------------------------------------------------------------
        2.00e-06 |     35.91% |      0.44% | [36.24%, 36.09%, 35.41%]
        3.56e-06 |     44.75% |      0.69% | [45.03%, 43.97%, 45.26%]
        6.32e-06 |     54.39% |      0.81% | [55.27%, 53.68%, 54.21%]
        1.12e-05 |     63.56% |      0.65% | [64.22%, 63.53%, 62.93%]
        2.00e-05 |     65.50% |      0.23% | [65.28%, 65.73%, 65.50%]
        3.56e-05 |     64.97% |      0.23% | [64.75%, 65.20%, 64.97%]
        6.32e-05 |     65.63% |      0.58% | [65.50%, 65.13%, 66.26%]
        1.12e-04 |     66.94% |      0.46% | [66.49%, 67.40%, 66.94%]
        2.00e-04 |     66.82% |      0.23% | [66.57%, 66.87%, 67.02%]
        3.56e-04 |     65.48% |      0.99% | [64.37%, 66.26%, 65.81%]
        6.32e-04 |     63.08% |      0.60% | [63.31%, 62.40%, 63.53%]
        1.12e-03 |     58.17% |      0.71% | [58.98%, 57.92%, 57.62%]
        2.00e-03 |     14.94% |     23.31% | [1.14%, 41.85%, 1.82%]

    ======================================================================
    SUB-TASK PERFORMANCE: MATH
    ======================================================================

    [DoRA]
            LR |       Mean |     StdDev | Runs
    ----------------------------------------------------------------------
        2.00e-06 |     31.07% |      0.19% | [31.14%, 31.22%, 30.86%]
        3.56e-06 |     31.29% |      0.61% | [31.84%, 31.38%, 30.64%]
        6.32e-06 |     32.18% |      0.19% | [32.20%, 31.98%, 32.36%]
        1.12e-05 |     32.25% |      0.11% | [32.34%, 32.28%, 32.12%]
        2.00e-05 |     32.84% |      0.27% | [32.86%, 33.10%, 32.56%]
        3.56e-05 |     33.10% |      0.40% | [32.90%, 32.84%, 33.56%]
        6.32e-05 |     32.87% |      0.55% | [33.26%, 33.12%, 32.24%]
        1.12e-04 |     33.07% |      0.27% | [33.32%, 32.78%, 33.12%]
        2.00e-04 |     32.77% |      0.39% | [32.32%, 33.00%, 33.00%]
        3.56e-04 |     31.69% |      0.83% | [32.64%, 31.10%, 31.34%]
        6.32e-04 |     28.45% |      0.40% | [28.38%, 28.08%, 28.88%]
        1.12e-03 |     20.97% |      0.50% | [21.20%, 21.32%, 20.40%]
        2.00e-03 |      4.41% |      6.16% | [11.52%, 0.82%, 0.88%]

    [LoRA]
            LR |       Mean |     StdDev | Runs
    ----------------------------------------------------------------------
        2.00e-06 |     25.74% |      0.34% | [25.70%, 26.10%, 25.42%]
        3.56e-06 |     27.21% |      0.16% | [27.14%, 27.40%, 27.10%]
        6.32e-06 |     31.25% |      0.16% | [31.14%, 31.44%, 31.18%]
        1.12e-05 |     32.61% |      0.27% | [32.84%, 32.68%, 32.32%]
        2.00e-05 |     32.83% |      0.33% | [32.76%, 32.54%, 33.18%]
        3.56e-05 |     32.96% |      0.11% | [32.88%, 32.92%, 33.08%]
        6.32e-05 |     32.93% |      0.52% | [33.44%, 32.96%, 32.40%]
        1.12e-04 |     33.42% |      0.66% | [34.18%, 33.10%, 32.98%]
        2.00e-04 |     32.95% |      0.28% | [33.06%, 32.64%, 33.16%]
        3.56e-04 |     31.87% |      0.29% | [32.18%, 31.60%, 31.82%]
        6.32e-04 |     28.20% |      0.11% | [28.24%, 28.08%, 28.28%]
        1.12e-03 |     21.52% |      1.09% | [21.18%, 20.64%, 22.74%]
        2.00e-03 |      1.19% |      0.23% | [1.06%, 1.06%, 1.46%]

    [InitAB]
            LR |       Mean |     StdDev | Runs
    ----------------------------------------------------------------------
        2.00e-06 |     30.52% |      0.32% | [30.20%, 30.52%, 30.84%]
        3.56e-06 |     31.36% |      0.39% | [31.00%, 31.30%, 31.78%]
        6.32e-06 |     32.30% |      0.19% | [32.32%, 32.10%, 32.48%]
        1.12e-05 |     32.77% |      0.25% | [32.58%, 33.06%, 32.68%]
        2.00e-05 |     33.35% |      0.14% | [33.48%, 33.20%, 33.36%]
        3.56e-05 |     32.62% |      0.56% | [32.34%, 32.26%, 33.26%]
        6.32e-05 |     32.13% |      0.39% | [31.90%, 32.58%, 31.92%]
        1.12e-04 |     32.17% |      0.26% | [32.14%, 32.44%, 31.92%]
        2.00e-04 |     30.93% |      0.25% | [31.18%, 30.68%, 30.94%]
        3.56e-04 |     27.75% |      0.29% | [27.54%, 28.08%, 27.62%]
        6.32e-04 |     25.33% |      0.40% | [25.16%, 25.04%, 25.78%]
        1.12e-03 |     19.43% |      0.60% | [19.48%, 18.80%, 20.00%]
        2.00e-03 |      1.03% |      0.17% | [1.10%, 0.84%, 1.16%]

    [PiSSA]
            LR |       Mean |     StdDev | Runs
    ----------------------------------------------------------------------
        2.00e-06 |     32.11% |      0.17% | [31.94%, 32.28%, 32.10%]
        3.56e-06 |     32.87% |      0.29% | [32.66%, 32.76%, 33.20%]
        6.32e-06 |     32.86% |      0.08% | [32.86%, 32.94%, 32.78%]
        1.12e-05 |     32.45% |      0.27% | [32.76%, 32.32%, 32.26%]
        2.00e-05 |     32.74% |      0.24% | [32.86%, 32.46%, 32.90%]
        3.56e-05 |     31.62% |      0.11% | [31.74%, 31.52%, 31.60%]
        6.32e-05 |     30.86% |      0.24% | [30.90%, 30.60%, 31.08%]
        1.12e-04 |     27.63% |      0.06% | [27.68%, 27.56%, 27.66%]
        2.00e-04 |     24.17% |      0.10% | [24.06%, 24.22%, 24.24%]
        3.56e-04 |     20.18% |      0.08% | [20.18%, 20.26%, 20.10%]
        6.32e-04 |     15.70% |      0.29% | [15.88%, 15.86%, 15.36%]
        1.12e-03 |     10.35% |      0.85% | [9.62%, 10.16%, 11.28%]
        2.00e-03 |      5.16% |      0.16% | [5.34%, 5.04%, 5.10%]

    [MiLoRA]
            LR |       Mean |     StdDev | Runs
    ----------------------------------------------------------------------
        2.00e-06 |     31.33% |      0.45% | [31.16%, 31.84%, 30.98%]
        3.56e-06 |     31.94% |      0.34% | [32.32%, 31.68%, 31.82%]
        6.32e-06 |     32.35% |      0.05% | [32.34%, 32.30%, 32.40%]
        1.12e-05 |     33.04% |      0.11% | [32.92%, 33.08%, 33.12%]
        2.00e-05 |     32.65% |      0.24% | [32.48%, 32.92%, 32.54%]
        3.56e-05 |     32.51% |      0.42% | [32.10%, 32.94%, 32.48%]
        6.32e-05 |     32.71% |      0.19% | [32.64%, 32.56%, 32.92%]
        1.12e-04 |     31.22% |      0.26% | [31.34%, 30.92%, 31.40%]
        2.00e-04 |     29.62% |      0.22% | [29.42%, 29.86%, 29.58%]
        3.56e-04 |     27.65% |      0.56% | [27.26%, 28.30%, 27.40%]
        6.32e-04 |     24.75% |      0.81% | [24.02%, 24.60%, 25.62%]
        1.12e-03 |     19.23% |      0.32% | [19.04%, 19.60%, 19.04%]
        2.00e-03 |      4.03% |      5.50% | [0.82%, 10.38%, 0.90%]

    ======================================================================
    AGGREGATED PERFORMANCE (per method)
    ======================================================================

    [DoRA]
            LR |       Mean |     StdDev | Runs
    ----------------------------------------------------------------------
        2.00e-06 |     36.84% |      0.86% | [36.23%, 36.47%, 37.82%]
        3.56e-06 |     38.19% |      1.74% | [38.05%, 36.51%, 39.99%]
        6.32e-06 |     39.29% |      0.36% | [39.68%, 38.96%, 39.23%]
        1.12e-05 |     44.72% |      0.96% | [45.44%, 43.62%, 45.09%]
        2.00e-05 |     48.09% |      0.74% | [48.05%, 48.85%, 47.37%]
        3.56e-05 |     49.01% |      0.44% | [49.49%, 48.91%, 48.64%]
        6.32e-05 |     49.25% |      0.33% | [49.58%, 49.27%, 48.91%]
        1.12e-04 |     49.45% |      0.25% | [49.18%, 49.68%, 49.50%]
        2.00e-04 |     49.33% |      0.45% | [49.82%, 49.21%, 48.95%]
        3.56e-04 |     49.32% |      0.71% | [48.92%, 48.91%, 50.13%]
        6.32e-04 |     46.92% |      0.28% | [46.64%, 47.20%, 46.93%]
        1.12e-03 |     40.24% |      0.62% | [39.62%, 40.85%, 40.26%]
        2.00e-03 |     10.14% |     15.64% | [28.20%, 1.12%, 1.09%]

    [LoRA]
            LR |       Mean |     StdDev | Runs
    ----------------------------------------------------------------------
        2.00e-06 |     21.48% |      0.53% | [22.04%, 20.97%, 21.41%]
        3.56e-06 |     24.52% |      0.44% | [24.43%, 25.00%, 24.14%]
        6.32e-06 |     31.94% |      0.58% | [31.98%, 32.50%, 31.34%]
        1.12e-05 |     43.07% |      1.16% | [42.73%, 42.11%, 44.36%]
        2.00e-05 |     48.37% |      0.19% | [48.45%, 48.15%, 48.51%]
        3.56e-05 |     48.81% |      0.14% | [48.84%, 48.95%, 48.66%]
        6.32e-05 |     49.27% |      0.34% | [49.42%, 49.51%, 48.88%]
        1.12e-04 |     49.46% |      0.56% | [49.08%, 50.10%, 49.20%]
        2.00e-04 |     49.60% |      0.18% | [49.60%, 49.41%, 49.78%]
        3.56e-04 |     48.95% |      0.20% | [48.97%, 48.74%, 49.15%]
        6.32e-04 |     47.08% |      0.22% | [46.98%, 47.33%, 46.93%]
        1.12e-03 |     40.76% |      0.96% | [41.85%, 40.04%, 40.38%]
        2.00e-03 |      1.27% |      0.15% | [1.18%, 1.19%, 1.44%]

    [InitAB]
            LR |       Mean |     StdDev | Runs
    ----------------------------------------------------------------------
        2.00e-06 |     33.40% |      2.16% | [30.92%, 34.60%, 34.70%]
        3.56e-06 |     35.81% |      1.72% | [34.19%, 37.61%, 35.63%]
        6.32e-06 |     41.20% |      1.83% | [39.10%, 42.47%, 42.02%]
        1.12e-05 |     47.27% |      0.89% | [48.30%, 46.71%, 46.80%]
        2.00e-05 |     49.27% |      0.13% | [49.34%, 49.12%, 49.35%]
        3.56e-05 |     49.02% |      0.11% | [48.97%, 49.15%, 48.96%]
        6.32e-05 |     48.81% |      0.17% | [48.96%, 48.81%, 48.63%]
        1.12e-04 |     49.29% |      0.23% | [49.50%, 49.32%, 49.06%]
        2.00e-04 |     48.51% |      0.44% | [48.01%, 48.64%, 48.88%]
        3.56e-04 |     47.37% |      0.39% | [46.94%, 47.48%, 47.70%]
        6.32e-04 |     44.81% |      0.34% | [44.97%, 44.42%, 45.04%]
        1.12e-03 |     39.41% |      0.49% | [39.61%, 39.77%, 38.85%]
        2.00e-03 |      0.93% |      0.41% | [0.66%, 0.73%, 1.41%]

    [PiSSA]
            LR |       Mean |     StdDev | Runs
    ----------------------------------------------------------------------
        2.00e-06 |     38.13% |      0.84% | [37.59%, 39.10%, 37.69%]
        3.56e-06 |     44.51% |      0.26% | [44.24%, 44.77%, 44.54%]
        6.32e-06 |     48.11% |      0.14% | [48.05%, 48.27%, 48.00%]
        1.12e-05 |     48.77% |      0.12% | [48.73%, 48.69%, 48.91%]
        2.00e-05 |     49.43% |      0.19% | [49.48%, 49.21%, 49.58%]
        3.56e-05 |     49.09% |      0.16% | [49.05%, 49.27%, 48.97%]
        6.32e-05 |     48.44% |      0.12% | [48.31%, 48.55%, 48.45%]
        1.12e-04 |     47.10% |      0.32% | [47.17%, 46.75%, 47.38%]
        2.00e-04 |     43.84% |      0.28% | [43.53%, 44.07%, 43.91%]
        3.56e-04 |     39.66% |      0.36% | [39.92%, 39.25%, 39.81%]
        6.32e-04 |     34.37% |      0.39% | [34.80%, 34.06%, 34.25%]
        1.12e-03 |     27.18% |      1.29% | [27.43%, 25.77%, 28.32%]
        2.00e-03 |     17.35% |      0.57% | [17.84%, 16.72%, 17.50%]

    [MiLoRA]
            LR |       Mean |     StdDev | Runs
    ----------------------------------------------------------------------
        2.00e-06 |     33.62% |      0.39% | [33.97%, 33.20%, 33.70%]
        3.56e-06 |     38.35% |      0.46% | [37.82%, 38.54%, 38.67%]
        6.32e-06 |     43.37% |      0.41% | [42.99%, 43.31%, 43.80%]
        1.12e-05 |     48.30% |      0.27% | [48.02%, 48.30%, 48.57%]
        2.00e-05 |     49.08% |      0.23% | [48.88%, 49.32%, 49.02%]
        3.56e-05 |     48.74% |      0.32% | [48.73%, 49.07%, 48.42%]
        6.32e-05 |     49.17% |      0.38% | [48.84%, 49.07%, 49.59%]
        1.12e-04 |     49.08% |      0.14% | [49.16%, 48.92%, 49.17%]
        2.00e-04 |     48.22% |      0.20% | [48.36%, 48.30%, 47.99%]
        3.56e-04 |     46.57% |      0.73% | [46.61%, 47.28%, 45.82%]
        6.32e-04 |     43.91% |      0.58% | [43.50%, 43.66%, 44.57%]
        1.12e-03 |     38.70% |      0.34% | [38.33%, 39.01%, 38.76%]
        2.00e-03 |      9.48% |     14.40% | [0.98%, 1.36%, 26.11%]

    ======================================================================
    BEST PERFORMANCE FOR EACH METHOD
    ======================================================================
    DoRA         | Best: 49.45% (±0.25%) | LR: 1.12e-04 | Runs: 3
    LoRA         | Best: 49.60% (±0.18%) | LR: 2.00e-04 | Runs: 3
    InitAB       | Best: 49.29% (±0.23%) | LR: 1.12e-04 | Runs: 3
    PiSSA        | Best: 49.43% (±0.19%) | LR: 2.00e-05 | Runs: 3
    MiLoRA       | Best: 49.17% (±0.38%) | LR: 6.32e-05 | Runs: 3

    ======================================================================
    RANKING (by Best Accuracy)
    ======================================================================
    1. LoRA         | 49.60% (±0.18%) | LR: 2.00e-04 | Runs: 3
    2. DoRA         | 49.45% (±0.25%) | LR: 1.12e-04 | Runs: 3
    3. PiSSA        | 49.43% (±0.19%) | LR: 2.00e-05 | Runs: 3
    4. InitAB       | 49.29% (±0.23%) | LR: 1.12e-04 | Runs: 3
    5. MiLoRA       | 49.17% (±0.38%) | LR: 6.32e-05 | Runs: 3
    ======================================================================


</details>

### 3. Run other model-task combinations or training setup

+ Please see script files organized under `./scripts` for example for running experiments on different model-task combinations.  
+ You can also easily set up learning rate search range, batch size, per-device batch size, etc, in the script.



## On Reproducibility

### Seeding

We understand people might be interested in reproducing our results. While we assign fixed random seeds across PyTorch, NumPy, and Python per [official Pytorch guidance](https://docs.pytorch.org/docs/2.11/notes/randomness.html):

```python
def seed_everything(seed):
    logger.info(f'===== Seed everything with seed:{seed} =====')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
```

We note there are several known sources of non-determinism that may affect results across different runs and devices — that is why we run three seeds and report means ± stds for all models smaller than 7B.
We document the known sources below; feel free to open issues to help us keep this list current.

### Sources of Randomness

#### 1. Numerical Precision
- Following prior work, we run base models at **BFloat16** precision and LoRA weights at **Float32**.
- BFloat16 is known to be more susceptible to numerical instability than Float32.

#### 2. During Training

#### 2.1 LoRA Weight Initialization

- *Vanilla LoRA* uses Kaiming Uniform initialization for `lora_A`. Each run draws fresh random values for `lora_A`, so results vary across runs unless `torch.manual_seed` is set.
- The same applies to *Gradient Modification* and *Optimization Adjustment* LoRA variants, and to *Init[AB]* methods within the *Initialization Variants* family.
- For other *initialization variants*, even though the method itself is meant to be deterministic given the same input:
    - QR-based (OLoRA):
        [`torch.linalg.qr`](https://pytorch.org/docs/stable/generated/torch.linalg.qr.html) explicitly warns:
        > *"The returned QR decomposition is only unique up to the sign of the diagonal of R. Therefore, different platforms, like NumPy, or inputs on different devices, may produce different valid decompositions."*
    - SVD-based (PiSSA, MiLoRA, LoRA-GA):
        [`torch.linalg.svd`](https://pytorch.org/docs/stable/generated/torch.linalg.svd.html) explicitly warns:
        > *"The returned tensors U and V are not unique, nor are they continuous with respect to A. Due to this lack of uniqueness, different hardware and software may compute different singular vectors."*

    - ***While we upcast base weights to Float32 before QR and SVD to improve numerical stability, cross-device or cross-platform uniqueness is still not guaranteed.***

#### 2.2. PEFT Version Sensitivity

- Different PEFT versions may produce slightly different results under identical hyperparameters, though the differences are typically negligible.


#### 3. During Inference

- We use **vLLM** for parallelized inference with temperature set to zero. 
- While this implies greedy decoding, [vLLM does not guarantee complete reproducibility by default](https://docs.vllm.ai/en/latest/usage/reproducibility/).

> [!NOTE]
> Despite these sources of randomness, the results remain representative. In our experience, the variance across runs with the same hyperparameters is much smaller than the performance gap induced by changing the learning rate or batch size.
> If after running multiple seeds and computing mean ± std, your results still deviate substantially from ours, please feel free to open an issue and share your results.

## Known Issues
Please feel free to open issues to report errors or package impatibility you faced.



