import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODELS = ["Qwen", "phi-tiny", "gemma"]

def get_num_layers(model):
    config = model.config
    for attr in ['num_hidden_layers', 'n_layer', 'n_layers', 'num_layers']:
        if hasattr(config, attr) and getattr(config, attr) is not None:
            return getattr(config, attr)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return len(model.model.layers)
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return len(model.transformer.h)
    if hasattr(model, 'layers'):
        return len(model.layers)
    max_layer = -1
    for name, _ in model.named_parameters():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            idx = int(match.group(1))
            if idx > max_layer:
                max_layer = idx
    if max_layer >= 0:
        return max_layer + 1

def compute_weight_metrics(model, layer_idx, compute_device):
    spectral_norms = []
    frob_norms = []
    eff_ranks = []
    layer_signature = f".{layer_idx}."
    for name, param in model.named_parameters():
        if layer_signature in name and len(param.shape) == 2:
            W = param.detach().to(compute_device).float()
            frob = torch.norm(W, p='fro').item()
            frob_norms.append(frob)
            try:
                S = torch.linalg.svdvals(W)
                spec = S[0].item()
                spectral_norms.append(spec)
                p = S / (S.sum() + 1e-9)
                ent = -torch.sum(p * torch.log(p + 1e-9))
                erank = torch.exp(ent).item()
                eff_ranks.append(erank)
            except Exception:
                pass
            del W
    return {
        'Spectral_Norm': np.mean(spectral_norms) if spectral_norms else 0.0,
        'Frobenius_Norm': np.mean(frob_norms) if frob_norms else 0.0,
        'Effective_Rank': np.mean(eff_ranks) if eff_ranks else 0.0
    }

def main():
    compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    for model_name in MODELS:
        model_path = os.path.join(MODELS_DIR, model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            device_map="cpu", 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        num_layers = get_num_layers(model)
        m_spec, m_frob, m_erank = [], [], []
        for layer_idx in range(num_layers):
            weight_metrics = compute_weight_metrics(model, layer_idx, compute_device)
            m_spec.append(weight_metrics['Spectral_Norm'])
            m_frob.append(weight_metrics['Frobenius_Norm'])
            m_erank.append(weight_metrics['Effective_Rank'])
        out_dir = os.path.join(SCRIPT_DIR, model_name)
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame({'Spectral_Norm': m_spec}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(out_dir, "metric_14_Spectral_Norm.csv"))
        pd.DataFrame({'Frobenius_Norm': m_frob}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(out_dir, "metric_15_Frobenius_Norm.csv"))
        pd.DataFrame({'Effective_Rank': m_erank}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(out_dir, "metric_16_Effective_Rank.csv"))
        del model
        if compute_device == "cuda":
            torch.cuda.empty_cache()

main()
