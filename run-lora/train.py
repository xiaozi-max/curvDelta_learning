import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import logging
import json
import os

import torch
import torch.distributed
import transformers
from transformers import Trainer, BitsAndBytesConfig
from datasets import load_dataset, concatenate_datasets
import datasets
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel, LoraRuntimeConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from utils.curv_lora import CurvLoRATrainer, apply_curv_lora
from utils.curv_delta import CurvDeltaTrainer, get_curv_delta_model
from utils.dense_peft import DenseConfig
from utils.crispedit import CrispEditTrainer
from utils.op_lora import OPLoRAFoldCallback, apply_op_lora

IGNORE_INDEX = -100
logger = logging.getLogger(__name__)

PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )


def seed_everything(seed):
    print(f'===== Seed everything with seed:{seed} =====')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # Seed
    trial_id: int = field(default=1, metadata={"help": "Seed for the experiment"})
    
    # Base model and adapter path ##
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B", metadata={"help": ("the path to base model, either pretrained weights or residaul weights"),},)
    adapter_name_or_path: Optional[str] = field(default=None,metadata={"help": ("the subfolder name in model_name_or_path where the pre-initialized lora weights is; when this is not None, the following `init_weights` argument will be ignored."),},)
    init_weights: bool | str = field(default=True,metadata={"help": ("True -> LoRA (Kaiming Uniform); `pissa` -> PiSSA; `pissa_niter_16` -> Fast SVD PiSSA"),},)
    
    # LoRA variants
    ## OLoRA
    use_olora : Optional[bool] = field(default=False)
    ## LoFT
    use_loft : Optional[bool] = field(default=False)
    ## Init[AB]
    use_initAB : Optional[bool] = field(default=False)
    initAB_res_model: Optional[str] = field(default=None,metadata={"help": ("path to save InitAB res model"),},)
    ## DoRA
    use_dora : Optional[bool] = field(default=False)    
    ## GraLoRA
    use_gralora : Optional[bool] = field(default=False)
    gralora_k: int = field(default=2, metadata={"help": "the architectural hyperparameter k of GraLoRA \
                    gralora_k=2 is recommended for rank 32 or lower, and gralora_k=4 is recommended for rank 64 or higher \
                    https://github.com/huggingface/peft/blob/758cdac51922abbb24b6e772844c0a88bbe1cd7d/src/peft/tuners/gralora/config.py#L119"})
    ## RandLoRA
    use_randlora : Optional[bool] = field(default=False)
    randlora_alpha: int = field(default=None, metadata={"help": "the alpha hyperparameter of RandLoRA \
                                typically 20 times the rank of the random bases. \
                                https://github.com/huggingface/peft/blob/758cdac51922abbb24b6e772844c0a88bbe1cd7d/src/peft/tuners/randlora/config.py#L140"})
    ## CurvLoRA
    use_curv_lora: Optional[bool] = field(default=False)
    curvature_data_path: Optional[str] = field(default=None, metadata={"help": "Path to raw-text JSONL curvature anchor set."})
    curvature_energy_ratio: float = field(default=0.9, metadata={"help": "KFAC top-eigenvector energy ratio for CurvLoRA projections."})
    curvature_collection_batch_size: int = field(default=1, metadata={"help": "Per-device batch size for CurvLoRA KFAC/Fisher collection."})
    curvature_num_chunks: int = field(default=1, metadata={"help": "Number of target-module chunks for CurvLoRA KFAC collection."})
    curvature_alpha_tr: float = field(default=0.0, metadata={"help": "Trust-region alpha: ||Delta W_eff||_F <= alpha_tr * ||W_base||_F. 0 = disabled."})
    curvature_delta_tr: float = field(default=1e-12, metadata={"help": "Numerical stabilizer for trust-region denominator (must be > 0)."})
    ## CurvDelta
    use_curv_delta: Optional[bool] = field(default=False)
    ## CrispEdit
    use_crispedit: Optional[bool] = field(default=False)
    crispedit_data_path: Optional[str] = field(default=None, metadata={"help": "Path to FineFineWeb JSONL (field 'text') used for the fixed CrispEdit K-FAC cache."})
    crispedit_energy_ratio: float = field(default=0.9, metadata={"help": "Kronecker-eigenvalue energy ratio for the CrispEdit low-curvature mask."})
    crispedit_collection_batch_size: int = field(default=1, metadata={"help": "Per-device batch size for CrispEdit K-FAC collection."})
    crispedit_num_chunks: int = field(default=1, metadata={"help": "Number of target-module chunks for CrispEdit K-FAC collection."})
    ## OPLoRA
    use_op_lora: Optional[bool] = field(default=False)
    op_lora_k: int = field(default=16, metadata={"help": "Number of top singular directions of W0 protected by OPLoRA."})
        
    # LoRA setting
    target_modules : Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_alpha : Optional[float] = field(default=32.)
    lora_dropout : Optional[float] = field(default=0.,metadata={"help": ("Must be set to 0 when using PiSSA."),},)
    
    # Training setting
    optim: str = field(default="adamw_torch")
    attn_implementation : Optional[str] = field(default="flash_attention_2")
    full_finetune : Optional[bool] = field(default=False)
    model_max_length: int = field(default=512,metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)    
    merge : Optional[bool] = field(default=False,metadata={"help": "Merge the adapters to the base weights"},)

    # Quantization setting
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    
    # DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    sub_task: List[str] = field(default=None)
    dataset_split: str = field(default="train", metadata={"help": "(`['train', 'test', 'eval']`):"})
    dataset_field: List[str] = field(default=None, metadata={"help": "Fields of dataset input and output."})
    shuffle_dataset: Optional[bool] = field(default=False)

    # Replay (pretraining-data mixing): optional raw-text JSONL (field "text")
    # mixed into the training set as next-token-prediction examples. Empty /
    # None = original behaviour (task data only).
    replay_data_path: Optional[str] = field(default=None, metadata={"help": "Raw-text JSONL (field 'text') of pretraining data to mix into the training set for replay-style training. None = original behaviour."})
    replay_num_samples: Optional[int] = field(default=None, metadata={"help": "Number of replay samples to take from replay_data_path. None = use all."})
    

class SavePeftModelCallback(transformers.TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def save_model(self, args, state, kwargs):
        logger.info('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        tokenizer = kwargs.get("tokenizer", self.tokenizer)
        tokenizer.save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        logger.info('Saving final PEFT model...')
        peft_model_path = os.path.join(args.output_dir, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        tokenizer = kwargs.get("tokenizer", self.tokenizer)
        tokenizer.save_pretrained(peft_model_path)
        
        # make a "completed" mark in the folder to denote that the training has completed       
        with open(os.path.join(args.output_dir, 'completed'), 'a'):
            pass


def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, 'completed'))
        if is_completed: return None  # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith(PREFIX_CHECKPOINT_DIR):
                max_step = max(max_step, int(filename.replace(PREFIX_CHECKPOINT_DIR + '-', '')))
        if max_step == 0: return None
        latest_ckpt_dir = os.path.join(checkpoint_dir, f'{PREFIX_CHECKPOINT_DIR}-{max_step}')
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return latest_ckpt_dir
    return None  # first training


def _is_zero3(trainer: transformers.Trainer) -> bool:
    """Return True when running under DeepSpeed ZeRO stage 3."""
    if not getattr(trainer, "is_deepspeed_enabled", False):
        return False
    try:
        plugin = getattr(getattr(trainer, "accelerator", None), "state", None)
        plugin = getattr(plugin, "deepspeed_plugin", None)
        if plugin is not None and getattr(plugin, "zero_stage", None) is not None:
            return int(plugin.zero_stage) == 3
    except Exception:
        pass
    return False


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk.

    Under DeepSpeed ZeRO-3, ``model.state_dict()`` returns only the local
    shard of every parameter (a 7B model would be saved as ~KB), producing a
    corrupt checkpoint that vLLM cannot load (shape asserts on vocab/embedding).
    In that case we gather the consolidated 16-bit weights across all ranks
    (a collective op) and only the main process writes them to disk.
    """
    if _is_zero3(trainer):
        engine = getattr(trainer, "model_wrapped", None) or getattr(trainer, "deepspeed", None)
        gather = getattr(engine, "_zero3_consolidated_16bit_state_dict", None)
        if callable(gather):
            state_dict = gather()  # collective — every rank must call it
            if trainer.args.should_save:
                cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
                del state_dict
                trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
            return
    if trainer.args.should_save:
        state_dict = trainer.model.state_dict()
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [tokenizer(text, max_length=tokenizer.model_max_length, truncation=True,) for text in strings]
    input_ids = labels = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [len(tokenized.input_ids) for tokenized in tokenized_list]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}\n{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict


def replay_tokenize_function(examples, tokenizer, model_max_length):
    """Tokenize raw pretraining text for next-token prediction.

    Replay data has no instruction/response split: the whole text is a
    continuation, so labels equal input_ids (padding is masked to IGNORE_INDEX
    by DataCollatorForSupervisedDataset).
    """
    texts = [t for t in examples.get("text", []) if isinstance(t, str) and t.strip()]
    tokenized = tokenizer(
        texts,
        max_length=model_max_length,
        truncation=True,
        padding=False,
    )
    return {"input_ids": tokenized["input_ids"], "labels": tokenized["input_ids"]}


def build_model(script_args, checkpoint_dir):
    if script_args.full_finetune:
        assert script_args.bits in [16, 32]
    compute_dtype = torch.bfloat16 if script_args.bf16 else torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=script_args.bits == 4,
            load_in_8bit=script_args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=script_args.double_quant,
            bnb_4bit_quant_type=script_args.quant_type,
        ) if script_args.bits in [4, 8] else None,
        dtype=compute_dtype,
        trust_remote_code=True,
        attn_implementation=script_args.attn_implementation,
    )
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    model.enable_input_require_grads()

    if not script_args.full_finetune and not script_args.use_crispedit:
        if script_args.bits < 16:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=script_args.gradient_checkpointing)

        if checkpoint_dir is not None:
            logger.info(f"Loading adapters from {checkpoint_dir}.")
            model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)

        elif script_args.adapter_name_or_path is not None:
            logger.info(f"Initilize adapters from {script_args.model_name_or_path}/{script_args.adapter_name_or_path}. (Deterministic)")
            model = PeftModel.from_pretrained(model, script_args.model_name_or_path,
                                              subfolder=script_args.adapter_name_or_path, is_trainable=True)

        else:
            if script_args.use_curv_delta:
                # CurvDelta uses its own DenseModel path — skip LoRA entirely.
                dense_config = DenseConfig(
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=script_args.target_modules.split(','),
                )
                model = get_curv_delta_model(model, dense_config)

            else:
                if script_args.use_gralora:
                    from peft import GraloraConfig
                    peft_config = GraloraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        target_modules=script_args.target_modules.split(','),
                        inference_mode=False,
                        r=script_args.lora_rank,
                        alpha=script_args.lora_alpha,
                        gralora_dropout=script_args.lora_dropout,
                        hybrid_r=0, # since we are not using the HybridGraLoRA method
                        gralora_k=script_args.gralora_k,
                    )

                elif script_args.use_randlora:
                    from peft import RandLoraConfig
                    peft_config = RandLoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        target_modules=script_args.target_modules.split(','),
                        inference_mode=False,
                        r=script_args.lora_rank,
                        randlora_alpha=script_args.randlora_alpha,
                        randlora_dropout=script_args.lora_dropout,
                    )

                else:
                    init_lora_weights = "olora" if script_args.use_olora else script_args.init_weights
                    peft_config = LoraConfig(
                        use_dora=script_args.use_dora,
                        runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=script_args.use_dora),
                        task_type=TaskType.CAUSAL_LM,
                        target_modules=script_args.target_modules.split(','),
                        inference_mode=False,
                        r=script_args.lora_rank,
                        lora_alpha=script_args.lora_alpha,
                        lora_dropout=script_args.lora_dropout,
                        init_lora_weights=init_lora_weights,
                    )
                    if script_args.use_olora:
                        # use GPU to perform QR decomposition will be faster
                        local_rank = int(os.environ.get("LOCAL_RANK", 0))
                        model = model.to(torch.device(f"cuda:{local_rank}"))

                model = get_peft_model(model, peft_config)

                if script_args.use_curv_lora:
                    model = apply_curv_lora(model)
                if script_args.use_op_lora:
                    model = apply_op_lora(model, k=script_args.op_lora_k)

                if script_args.use_olora:
                    olora_residual_path = os.path.join(script_args.output_dir, "olora_residual_model")
                    model_copy = copy.deepcopy(model)
                    residual = model_copy.unload()
                    residual.save_pretrained(olora_residual_path)
                    tok = transformers.AutoTokenizer.from_pretrained(script_args.model_name_or_path)
                    tok.pad_token_id = tok.eos_token_id
                    tok.save_pretrained(olora_residual_path)
                    logger.info(f"OLoRA residual model saved to {olora_residual_path}")

                if script_args.use_initAB:
                    from utils.init_AB_util import modify_initAB_model
                    modify_initAB_model(model, "AB_1_1_RESET")
                    model_copy = copy.deepcopy(model)
                    res_model = model_copy.unload()
                    res_model.save_pretrained(script_args.initAB_res_model)
                    tok = transformers.AutoTokenizer.from_pretrained(script_args.model_name_or_path)
                    tok.pad_token_id = tok.eos_token_id
                    tok.save_pretrained(script_args.initAB_res_model)
                    logger.info(f"InitAB residual model saved to {script_args.initAB_res_model}")

    for name, module in model.named_modules():
        if 'norm' in name or 'gate' in name:
            module = module.to(torch.float32)

    if script_args.local_rank == 0 and not script_args.use_curv_delta and not script_args.full_finetune and not script_args.use_crispedit:
        model.print_trainable_parameters()

    return model


def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    if "wandb" in script_args.report_to:
        parts = script_args.output_dir.rstrip('/').split('/')
        run_name = '/'.join(parts[-2:])
        os.environ["WANDB_PROJECT"] = "lr-matters-lora"
        os.environ["WANDB_NAME"] = run_name
    log_level = script_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    seed_everything(script_args.trial_id)

    if script_args.local_rank == 0:
        logger.info('=' * 100)
        logger.info(script_args)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if script_args.local_rank == 0:
        logger.info("Load tokenizer from {} over.".format(script_args.model_name_or_path))

    resume_from_checkpoint_dir = get_last_checkpoint(script_args.output_dir)
    if script_args.use_curv_lora and script_args.use_curv_delta:
        raise ValueError("Cannot use both CurvLoRA and CurvDelta simultaneously.")
    if script_args.use_op_lora:
        if script_args.full_finetune or script_args.use_curv_lora or script_args.use_curv_delta or script_args.use_crispedit or script_args.use_dora:
            raise ValueError("OPLoRA cannot be combined with FFT/CurvLoRA/CurvDelta/CrispEdit/DoRA.")
        if resume_from_checkpoint_dir is not None:
            raise ValueError("OPLoRA does not support resuming from unfinished checkpoints.")
        if script_args.op_lora_k <= 0:
            raise ValueError("--op_lora_k must be a positive integer.")
        if str(script_args.save_strategy).lower().split(".")[-1] != "no":
            raise ValueError("OPLoRA requires --save_strategy no; only the final folded adapter is saved.")
    if script_args.use_curv_lora:
        if resume_from_checkpoint_dir is not None:
            raise ValueError("CurvLoRA does not support resuming from unfinished checkpoints.")
        if script_args.curvature_data_path is None:
            raise ValueError("--curvature_data_path is required when --use_curv_lora True.")
        if str(script_args.save_strategy).lower().split(".")[-1] != "no":
            raise ValueError("CurvLoRA requires --save_strategy no; only the final folded adapter is saved.")
    if script_args.use_curv_delta:
        if resume_from_checkpoint_dir is not None:
            raise ValueError("CurvDelta does not support resuming from unfinished checkpoints.")
        if script_args.curvature_data_path is None:
            raise ValueError("--curvature_data_path is required when --use_curv_delta True.")
        if str(script_args.save_strategy).lower().split(".")[-1] != "no":
            raise ValueError("CurvDelta requires --save_strategy no; only the final folded model is saved.")
    if script_args.use_crispedit:
        if script_args.full_finetune or script_args.bits not in [16, 32]:
            raise ValueError("CrispEdit trains base projection weights in place: requires --full_finetune False and --bits 16/32.")
        if script_args.use_curv_lora or script_args.use_curv_delta or script_args.use_loft:
            raise ValueError("CrispEdit cannot be combined with CurvLoRA/CurvDelta/LoFT.")
        if resume_from_checkpoint_dir is not None:
            raise ValueError("CrispEdit does not support resuming from unfinished checkpoints.")
        if script_args.crispedit_data_path is None:
            raise ValueError("--crispedit_data_path is required when --use_crispedit True.")
        if str(script_args.save_strategy).lower().split(".")[-1] != "no":
            raise ValueError("CrispEdit requires --save_strategy no.")
        if str(script_args.lr_scheduler_type).lower().split(".")[-1] != "cosine":
            raise ValueError("CrispEdit requires --lr_scheduler_type cosine.")
    model = build_model(script_args, resume_from_checkpoint_dir)

    all_training_dataset = []
    for task in script_args.sub_task:
        if "commonsense" in task:
            from datasets import Dataset
            commonsense_path = f"{script_args.data_path}/commonsense_15k.json"
            with open(commonsense_path, 'r', encoding='utf-8') as f:
                commonsense_data = json.load(f)
            ds = Dataset.from_list(commonsense_data)
        else:
            if ":" in task:
                cur_task, num_split = task.split(":")
                cur_split = f"{script_args.dataset_split}[:{num_split}]"
            else:
                cur_task, cur_split = task, script_args.dataset_split
            ds = load_dataset(script_args.data_path, data_dir=cur_task, split=cur_split)
        if script_args.local_rank == 0:
            print(f"{script_args.data_path}/{task}/{ds.num_rows}")
            for k, v in ds[0].items():
                print("-" * 100)
                print(k, end=':\t')
                print(v)
            print("+" * 100)
        all_training_dataset.append(ds)

    raw_train_datasets = concatenate_datasets(all_training_dataset)
    if script_args.shuffle_dataset or script_args.num_train_epochs > 1:
        if script_args.local_rank == 0:
            print(f"Shuffle dataset with seed={script_args.seed}!")
        raw_train_datasets = raw_train_datasets.shuffle(seed=script_args.seed)

    if script_args.local_rank > 0:
        torch.distributed.barrier()

    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer, "query": script_args.dataset_field[0], "response": script_args.dataset_field[1]}
    )

    if script_args.local_rank == 0:
        torch.distributed.barrier()
        print(model)
        logger.info("Training dataset samples:", len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
            logger.info(f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")

    # Optional replay mixing: tokenize raw pretraining text (next-token
    # prediction), concatenate with the task dataset, then shuffle so the two
    # streams interleave (independent mixing — no per-sequence concatenation).
    if script_args.replay_data_path:
        replay_raw = load_dataset(
            "json", data_files=script_args.replay_data_path, split="train"
        )
        n_available = len(replay_raw)
        n_replay = (
            n_available
            if script_args.replay_num_samples is None
            else min(int(script_args.replay_num_samples), n_available)
        )
        if n_replay == 0:
            raise ValueError(
                f"Replay data '{script_args.replay_data_path}' selected 0 samples."
            )
        replay_raw = replay_raw.select(range(n_replay))
        if script_args.local_rank == 0:
            logger.info(
                "Replay: mixing %d/%d samples from %s",
                n_replay, n_available, script_args.replay_data_path,
            )
        # Rank 0 builds the (possibly fresh) Arrow cache first; other ranks wait.
        if script_args.local_rank > 0:
            torch.distributed.barrier()
        replay_dataset = replay_raw.map(
            replay_tokenize_function,
            batched=True,
            batch_size=3000,
            num_proc=32,
            remove_columns=replay_raw.column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on replay dataset",
            fn_kwargs={
                "tokenizer": tokenizer,
                "model_max_length": script_args.model_max_length,
            },
        )
        if script_args.local_rank == 0:
            torch.distributed.barrier()
        train_dataset = concatenate_datasets([train_dataset, replay_dataset])
        train_dataset = train_dataset.shuffle(seed=script_args.seed)
        if script_args.local_rank == 0:
            logger.info(
                "Replay mixing: task=%d + replay=%d = %d samples",
                len(train_dataset) - n_replay, n_replay, len(train_dataset),
            )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    if script_args.use_curv_lora:
        if script_args.use_loft:
            raise ValueError("CurvLoRA cannot be combined with LoFT optimizer mode.")
        trainer = CurvLoRATrainer(
            model=model,
            tokenizer=tokenizer,
            args=script_args,
            curvature_data_path=script_args.curvature_data_path,
            curvature_energy_ratio=script_args.curvature_energy_ratio,
            curvature_collection_batch_size=script_args.curvature_collection_batch_size,
            curvature_num_chunks=script_args.curvature_num_chunks,
            curvature_alpha_tr=script_args.curvature_alpha_tr,
            curvature_delta_tr=script_args.curvature_delta_tr,
            **data_module,
        )
    elif script_args.use_curv_delta:
        if script_args.use_loft:
            raise ValueError("CurvDelta cannot be combined with LoFT optimizer mode.")
        trainer = CurvDeltaTrainer(
            model=model,
            tokenizer=tokenizer,
            args=script_args,
            curvature_data_path=script_args.curvature_data_path,
            curvature_energy_ratio=script_args.curvature_energy_ratio,
            curvature_collection_batch_size=script_args.curvature_collection_batch_size,
            curvature_num_chunks=script_args.curvature_num_chunks,
            **data_module,
        )
    elif script_args.use_crispedit:
        if script_args.use_loft:
            raise ValueError("CrispEdit cannot be combined with LoFT optimizer mode.")
        trainer = CrispEditTrainer(
            model=model,
            tokenizer=tokenizer,
            args=script_args,
            target_modules=script_args.target_modules,
            crispedit_data_path=script_args.crispedit_data_path,
            crispedit_energy_ratio=script_args.crispedit_energy_ratio,
            crispedit_collection_batch_size=script_args.crispedit_collection_batch_size,
            crispedit_num_chunks=script_args.crispedit_num_chunks,
            **data_module,
        )
    elif script_args.use_loft:
        from utils.LoFT.utils import get_optimizer
        optimizer = get_optimizer(model, lr=script_args.learning_rate, LoFT=True)
        loft_max_norm = script_args.max_grad_norm if script_args.max_grad_norm is not None else 1.0
        script_args.max_grad_norm = None
        trainer = Trainer(model=model, tokenizer=tokenizer, args=script_args,
                          optimizers=(optimizer, None), **data_module)
        if loft_max_norm > 0:
            from utils.LoFT.loft_optim.hf_callbacks import GradientClippingCallback
            trainer.add_callback(GradientClippingCallback(max_norm=loft_max_norm))
    # elif script_args.adapter_name_or_path == 'loraga_init':
    #     trainer = Trainer(model=model, processing_class=tokenizer, args=script_args, **data_module)
    else:
        trainer = Trainer(model=model, tokenizer=tokenizer, args=script_args, **data_module)
    if script_args.use_op_lora:
        trainer.add_callback(OPLoRAFoldCallback())
    if not script_args.full_finetune and not script_args.use_curv_delta and not script_args.use_crispedit:
        trainer.add_callback(SavePeftModelCallback(tokenizer))
    trainer.train(resume_from_checkpoint=resume_from_checkpoint_dir)
    trainer.save_state()

    if not script_args.full_finetune:
        loss_file_path = os.path.join(script_args.output_dir, "train_loss.json")
        os.makedirs(os.path.dirname(loss_file_path), exist_ok=True)
        training_logs = [
            log for log in trainer.state.log_history
            if "loss" in log
        ]
        with open(loss_file_path, 'w', encoding='utf-8') as f:
            json.dump(training_logs, f, indent=2)
        logger.info(f'Training losses saved to {loss_file_path}')

    if not script_args.full_finetune and script_args.merge:
        model = model.merge_and_unload()
        model.save_pretrained(script_args.output_dir)
        tokenizer.save_pretrained(script_args.output_dir)
    if script_args.full_finetune:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=script_args.output_dir)
        # Save tokenizer alongside the full model for vLLM inference.
        tokenizer.save_pretrained(script_args.output_dir)
        # Write a "completed" mark so run_best_common.sh can detect finished FFT runs.
        with open(os.path.join(script_args.output_dir, 'completed'), 'a'):
            pass
        # Also save training loss for FFT.
        loss_file_path = os.path.join(script_args.output_dir, "train_loss.json")
        training_logs = [
            log for log in trainer.state.log_history
            if "loss" in log
        ]
        with open(loss_file_path, 'w', encoding='utf-8') as f:
            json.dump(training_logs, f, indent=2)
        logger.info(f'Training losses saved to {loss_file_path}')


if __name__ == "__main__":
    train()