import os
import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, logging
from tqdm import tqdm
import math
import warnings

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_NAME = "phi-tiny"  # 'gemma', 'qwen', 'phi-tiny'
TASK_NAME = "csqa"       # 'csqa', 'siqa', 'copa'
NUM_TEXTS = 50

MODEL_DIR = os.path.join(BASE_DIR, "models", MODEL_NAME)

def get_layers(model):
    if hasattr(model, 'language_model'):
        model = model.language_model
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    if hasattr(model, 'layers'):
        return model.layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 5:
            if 'vision' not in name.lower():
                return module

def get_skip_layer_hook():
    def hook(module, args, output):
        hidden_states = args[0]
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states
    return hook

def calculate_perplexity(model, tokenizer, texts, desc="Расчет PPL"):
    total_loss = 0.0
    total_length = 0
    loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for text in tqdm(texts, desc=desc, leave=False):
            inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True).to(model.device)
            if inputs["input_ids"].size(1) < 2:
                continue
            outputs = model(**inputs, use_cache=False)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()
            total_length += shift_labels.size(1)
    if total_length == 0:
        return float('inf')
    return math.exp(total_loss / total_length)

def load_texts(task_name):
    texts = []
    if task_name == "csqa":
        path = os.path.join(BASE_DIR, "datasets", "dev_rand_split.jsonl")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if isinstance(data, dict) and 'question' in data and 'stem' in data['question']:
                    texts.append(data['question']['stem'])
                if len(texts) >= NUM_TEXTS:
                    break
    elif task_name == "siqa":
        path = os.path.join(BASE_DIR, "datasets", "siqa_500.csv")
        df = pd.read_csv(path)
        text_col = 'context' if 'context' in df.columns else df.columns[0]
        texts = df[text_col].dropna().head(NUM_TEXTS).tolist()
    elif task_name == "copa":
        path = os.path.join(BASE_DIR, "datasets", "copa_500.csv")
        df = pd.read_csv(path)
        for _, row in df.head(NUM_TEXTS).iterrows():
            ans_col = row['answer'] if row['answer'] in ['A', 'B'] else 'A'
            texts.append(f"{row['question']} {row[ans_col]}")
    return texts

def main():
    texts = load_texts(TASK_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
    
    if MODEL_NAME.lower() == "qwen":
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, 
            device_map={"": 0},
            quantization_config=quant_config,
            local_files_only=True,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, 
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            local_files_only=True,
            trust_remote_code=True
        )
        
    model.eval()
    model.config.use_cache = False

    layers = get_layers(model)
    num_layers = len(layers)

    baseline_ppl = calculate_perplexity(model, tokenizer, texts)
    print(f"[{MODEL_NAME} | {TASK_NAME}] Базовая PPL: {baseline_ppl:.4f}\n")

    results = []
    for l in tqdm(range(num_layers), desc=f"Аблация {MODEL_NAME}"):
        hook_handle = layers[l].register_forward_hook(get_skip_layer_hook())
        ablated_ppl = calculate_perplexity(model, tokenizer, texts)
        hook_handle.remove()
        
        ppl_degradation = ablated_ppl - baseline_ppl
        results.append({
            "Layer": l,
            "Baseline_PPL": baseline_ppl,
            "Ablated_PPL": ablated_ppl,
            "PPL_Degradation": ppl_degradation
        })

    df_results = pd.DataFrame(results)
    out_file = f"ppl_drops_{MODEL_NAME.lower()}_{TASK_NAME}.csv"
    out_path = os.path.join(RESULTS_DIR, out_file)
    df_results.to_csv(out_path, index=False)
    print(f"Сохранено в: {out_path}")

if __name__ == "__main__":
    main()
