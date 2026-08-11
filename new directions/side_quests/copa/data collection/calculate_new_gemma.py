import os
import torch
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "copa_500.csv")

MODELS = ["gemma"]
DEVICE = "cuda"

def get_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'): return model.model.layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'): return model.transformer.h
    if hasattr(model, 'layers'): return model.layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 5:
            if 'vision' not in name.lower():
                return module

def get_num_layers(model):
    return len(get_layers(model))

def extract_m17_m18(model, tokenizer, df):
    num_layers = get_num_layers(model)
    all_hiddens = {l: [] for l in range(num_layers)}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  [Скрытые состояния для M17/M18]"):
        text = row['question'] + " " + row[row['answer']]
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            h_states = outputs.hidden_states[1:] 
            for l in range(num_layers):
                pooled = h_states[l].mean(dim=1).squeeze().cpu()
                all_hiddens[l].append(pooled)
    m17_var_shift = np.zeros(num_layers)
    m18_outliers = np.zeros(num_layers)

    for l in range(num_layers):
        h_matrix = torch.stack(all_hiddens[l]).float()
        token_vars = torch.var(h_matrix, dim=-1)
        m17_var_shift[l] = torch.std(token_vars).item()
        means = torch.mean(h_matrix, dim=0)
        stds = torch.std(h_matrix, dim=0) + 1e-9
        z_scores = torch.abs((h_matrix - means) / stds)
        outlier_mask = z_scores > 3.0
        m18_outliers[l] = torch.mean(outlier_mask.float()).item()
    return m17_var_shift, m18_outliers

def extract_m19(model, tokenizer, df):
    num_layers = get_num_layers(model)
    layers = get_layers(model)
    
    metrics_data = {
        'attn_norm': {i: [] for i in range(num_layers)},
        'mlp_norm': {i: [] for i in range(num_layers)}
    }
    def get_attn_hook(layer_idx):
        def hook(module, inp, output):
            out = output[0].float() if isinstance(output, tuple) else output.float()
            metrics_data['attn_norm'][layer_idx].append(out.norm(dim=-1).mean().item())
        return hook
    def get_mlp_hook(layer_idx):
        def hook(module, inp, output):
            out = output[0].float() if isinstance(output, tuple) else output.float()
            metrics_data['mlp_norm'][layer_idx].append(out.norm(dim=-1).mean().item())
        return hook
    hooks = []
    for i, layer in enumerate(layers):
        if hasattr(layer, 'self_attn'):
            hooks.append(layer.self_attn.register_forward_hook(get_attn_hook(i)))
        if hasattr(layer, 'mlp'):
            hooks.append(layer.mlp.register_forward_hook(get_mlp_hook(i)))
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  [M19 Attn vs MLP Ratio]"):
        text = row['question'] + " " + row[row['answer']]
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(**inputs, use_cache=False)
    for h in hooks:
        h.remove()

    m19_ratio = np.zeros(num_layers)
    for i in range(num_layers):
        avg_attn = np.mean(metrics_data['attn_norm'][i]) if metrics_data['attn_norm'][i] else 0.0
        avg_mlp = np.mean(metrics_data['mlp_norm'][i]) if metrics_data['mlp_norm'][i] else 0.0
        m19_ratio[i] = (avg_attn / (avg_mlp + 1e-6)) if avg_mlp > 0 else 0.0

    return m19_ratio

def main():
    df = pd.read_csv(DATASET_PATH)

    for model_name in MODELS:
        model_path = os.path.join(MODELS_DIR, model_name)
        dir_met = os.path.join(SCRIPT_DIR, "metrics", model_name)
        os.makedirs(dir_met, exist_ok=True)

        print(f"Загрузка модели {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            device_map="auto",
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True
        ).eval()
        model.config.use_cache = False

        if hasattr(model, 'language_model'):
            model = model.language_model
        
        num_layers = get_num_layers(model)

        m17, m18 = extract_m17_m18(model, tokenizer, df)
        
        pd.DataFrame({'Variance_Shift': m17}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(dir_met, "metric_17_Variance_shift.csv"))
        pd.DataFrame({'Outlier_IoU': m18}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(dir_met, "metric_18_Outliner_driven_signals.csv"))

        m19 = extract_m19(model, tokenizer, df)        
        pd.DataFrame({'Attn_vs_MLP_Ratio': m19}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(dir_met, "metric_19_Att_vs_MLP_ratio.csv"))
        
        del model, tokenizer
        if DEVICE == "cuda": torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
