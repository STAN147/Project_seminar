import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from tqdm import tqdm

logging.set_verbosity_error()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

PHI_MODEL_DIR = os.path.join(BASE_DIR, "models", "phi-tiny")
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "dev_rand_split.jsonl")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_LAYERS = 32
NOISE_STD = 0.1
BATCH_SIZE = 4
NUM_BATCHES = 4

def add_noise_pre_hook(std):
    def pre_hook(module, args):
        hidden_states = args[0]
        noise = torch.randn_like(hidden_states) * std
        noisy_hidden = hidden_states + noise
        return (noisy_hidden,) + args[1:]
    return pre_hook

def run_kl_analysis():
    tokenizer = AutoTokenizer.from_pretrained(PHI_MODEL_DIR, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        PHI_MODEL_DIR, 
        device_map={"": 0},
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True
    )
    model.eval()
    texts = []
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if 'question' in data and 'stem' in data['question']:
                texts.append(data['question']['stem'])
            else:
                texts.append(str(data))
            if len(texts) >= BATCH_SIZE * NUM_BATCHES:
                break
    kl_divergences = {i: [] for i in range(NUM_LAYERS)}
    with torch.no_grad():
        for b_idx in tqdm(range(NUM_BATCHES)):
            batch_texts = texts[b_idx*BATCH_SIZE : (b_idx+1)*BATCH_SIZE]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs_clean = model(**inputs, use_cache=False)
            logits_clean = outputs_clean.logits[:, -1, :] 
            prob_clean = F.softmax(logits_clean, dim=-1)
            for layer_idx in range(NUM_LAYERS):
                layer_module = model.model.layers[layer_idx]
                hook_handle = layer_module.register_forward_pre_hook(add_noise_pre_hook(NOISE_STD))
                outputs_noisy = model(**inputs, use_cache=False)
                logits_noisy = outputs_noisy.logits[:, -1, :]
                log_prob_noisy = F.log_softmax(logits_noisy, dim=-1)
                kl_div = F.kl_div(log_prob_noisy, prob_clean, reduction="batchmean")
                kl_divergences[layer_idx].append(kl_div.item())
                hook_handle.remove()
    final_metrics = []
    for layer_idx in range(NUM_LAYERS):
        mean_kl = sum(kl_divergences[layer_idx]) / len(kl_divergences[layer_idx])
        final_metrics.append({"Layer": layer_idx, "KL_Div_Noise": mean_kl})
    df_res = pd.DataFrame(final_metrics)
    df_res.set_index("Layer", inplace=True)
    out_path = os.path.join(RESULTS_DIR, "metric_12_KL_noise_csqa.csv")
    df_res.to_csv(out_path)

run_kl_analysis()
