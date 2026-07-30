import os
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, logging
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
logging.set_verbosity_error()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

QWEN_MODEL_DIR = os.path.join(BASE_DIR, "models", "Qwen")
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "siqa_500.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_LAYERS = 24
BATCH_SIZE = 4
NUM_BATCHES = 16

def run_logit_lens_analysis():
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_DIR, local_files_only=True)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_DIR, 
        device_map={"": 0},
        quantization_config=quant_config,
        local_files_only=True
    )
    model.eval()
    df_data = pd.read_csv(DATASET_PATH)
    text_col = 'context' if 'context' in df_data.columns else df_data.columns[0]
    texts = df_data[text_col].dropna().head(BATCH_SIZE * NUM_BATCHES).tolist()
    logit_lens_kl = {i: [] for i in range(NUM_LAYERS)}
    top1_match_rate = {i: [] for i in range(NUM_LAYERS)}
    with torch.no_grad():
        for b_idx in tqdm(range(NUM_BATCHES)):
            batch_texts = texts[b_idx*BATCH_SIZE : (b_idx+1)*BATCH_SIZE]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            final_logits = outputs.logits[:, -1, :] 
            final_probs = F.softmax(final_logits, dim=-1)
            final_preds = torch.argmax(final_logits, dim=-1)
            if hasattr(model.model, 'norm'):
                final_norm_func = model.model.norm
            elif hasattr(model.model, 'final_layernorm'):
                final_norm_func = model.model.final_layernorm
            else:
                final_norm_func = lambda x: x
            for layer_idx in range(NUM_LAYERS):
                intermediate_hidden = outputs.hidden_states[layer_idx + 1][:, -1, :]
                hidden_norm = final_norm_func(intermediate_hidden)
                intermediate_logits = model.lm_head(hidden_norm)
                intermediate_log_probs = F.log_softmax(intermediate_logits, dim=-1)
                kl_div = F.kl_div(intermediate_log_probs, final_probs, reduction="batchmean")
                logit_lens_kl[layer_idx].append(kl_div.item())
                intermediate_preds = torch.argmax(intermediate_logits, dim=-1)
                match_rate = (intermediate_preds == final_preds).float().mean().item()
                top1_match_rate[layer_idx].append(match_rate)
    final_metrics = []
    for layer_idx in range(NUM_LAYERS):
        mean_kl = sum(logit_lens_kl[layer_idx]) / len(logit_lens_kl[layer_idx])
        mean_match = sum(top1_match_rate[layer_idx]) / len(top1_match_rate[layer_idx])
        final_metrics.append({
            "Layer": layer_idx, 
            "Logit_Lens_KL": mean_kl,
            "Top1_Match_Rate": mean_match
        })
    df_res = pd.DataFrame(final_metrics)
    df_res.set_index("Layer", inplace=True)
    out_path = os.path.join(RESULTS_DIR, "metric_13_LogitLens_siqa.csv")
    df_res.to_csv(out_path)

run_logit_lens_analysis()
