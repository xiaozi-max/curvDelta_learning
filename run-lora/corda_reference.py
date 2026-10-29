import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.config import CordaConfig
from peft.tuners.lora.corda import preprocess_corda

model_path = "google/gemma-3-1b-pt"  # Replace with the local or Hub model path.
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ---- 1. Prepare calibration data (KPM uses world-knowledge data). ----
# Reuse NQ-Open questions to match the world-knowledge evaluation data.
calib_raw = load_dataset("nq_open", split="train").shuffle(seed=42).select(range(256))
calib_texts = [f"Question: {ex['question']}?\nAnswer:" for ex in calib_raw]

def run_model():
    # CorDA hooks capture the input activations of each linear layer, so a
    # regular forward pass is sufficient; no manual covariance is required.
    model.eval()
    with torch.no_grad():
        for text in calib_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            model(**inputs)

# ---- 2. Configure CorDA. ----
corda_config = CordaConfig(
    corda_method="kpm",              # "kpm" preserves knowledge; "ipm" favors the downstream task.
    # cache_file="corda_svd_cache.pt",       # Optional: reuse the SVD across ranks for one base model.
    # covariance_file="corda_cov.pt",        # Optional: reuse covariance with the same calibration data.
    # use_float16_for_covariance=True,       # Optional: saves memory but may be less stable than fp32.
    verbose=True,
)

lora_config = LoraConfig(
    r=128,                            # Match the experimental rank for fair baseline comparison.
    lora_alpha=128,
    target_modules="all-linear",      # Or use the same target modules as the other PEFT baselines.
    init_lora_weights="corda",
    corda_config=corda_config,
    task_type="CAUSAL_LM",
)

# ---- 3. Collect covariance and compute the SVD before get_peft_model. ----
preprocess_corda(model, lora_config, run_model=run_model)

# ---- 4. Wrap the PEFT model; subsequent training is the same as LoRA. ----
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# Continue with Trainer/SFTTrainer as in the LoRA and PiSSA launchers.
