import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODELS = ["gemma"]

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

def compute_layer_metrics(model, layer_idx, compute_device, is_moe):
    layer_signature = f".{layer_idx}."
    
    # Списки для базовых метрик
    spectral_norms, frob_norms, eff_ranks = [], [] , []
    
    # Контейнеры для M10 и M11
    router_metrics = {
        "Router_Norm_Var": 0.0,
        "Router_Norm_Min": 0.0,
        "Router_Norm_Max": 0.0,
        "Router_Norm_Mean": 0.0
    }
    svd_entropy_o_proj = 0.0
    
    # Флаги, чтобы взять только первый найденный тензор (как в исходном коде)
    router_found = False
    svd_found = False
    
    for name, param in model.named_parameters():
        if layer_signature in name and "weight" in name and len(param.shape) == 2:
            W = param.detach().to(compute_device).float()
            
            # --- Базовые метрики (M14, M15, M16) ---
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
                
                # --- M11: SVD Entropy O_Proj ---
                if not svd_found and ("o_proj" in name or "out_proj" in name or "dense" in name) and ("attn" in name or "mixer" in name):
                    max_entropy = torch.log(torch.tensor(len(S), dtype=torch.float32))
                    svd_entropy_o_proj = (ent / max_entropy).item()
                    svd_found = True
            except Exception:
                pass
                
            # --- M10: Router Weights (Только если модель MoE) ---
            if is_moe and not router_found and ("gate" in name or "router" in name) and "gate_proj" not in name:
                norms = torch.norm(W, p=2, dim=1)
                if len(norms) > 1:
                    router_metrics["Router_Norm_Var"] = torch.var(norms).item()
                router_metrics["Router_Norm_Min"] = torch.min(norms).item()
                router_metrics["Router_Norm_Max"] = torch.max(norms).item()
                router_metrics["Router_Norm_Mean"] = torch.mean(norms).item()
                router_found = True
                
            del W

    return {
        'Spectral_Norm': np.mean(spectral_norms) if spectral_norms else 0.0,
        'Frobenius_Norm': np.mean(frob_norms) if frob_norms else 0.0,
        'Effective_Rank': np.mean(eff_ranks) if eff_ranks else 0.0,
        'SVD_Entropy_O_Proj': svd_entropy_o_proj,
        **router_metrics
    }

def main():
    compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for model_name in MODELS:
        model_path = os.path.join(MODELS_DIR, model_name)
        
        if not os.path.exists(model_path):
            print(f"Путь не найден, пропускаем: {model_path}")
            continue
            
        print("="*60)
        print(f"Обработка модели: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            device_map="cpu", 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Детектор архитектуры (MoE или Dense) с исключением gate_proj
        is_moe = any(("gate" in name or "router" in name) and "gate_proj" not in name for name, _ in model.named_parameters())
        arch_type = "MoE (Метрики роутера будут вычислены)" if is_moe else "Dense (Метрики роутера заполнятся нулями)"
        print(f"Архитектура: {arch_type}")
        
        num_layers = get_num_layers(model)
        
        m_spec, m_frob, m_erank = [], [], []
        m_svd, m_router = [], []
        
        for layer_idx in range(num_layers):
            metrics = compute_layer_metrics(model, layer_idx, compute_device, is_moe)
            
            m_spec.append(metrics['Spectral_Norm'])
            m_frob.append(metrics['Frobenius_Norm'])
            m_erank.append(metrics['Effective_Rank'])
            m_svd.append(metrics['SVD_Entropy_O_Proj'])
            
            m_router.append({
                "Router_Norm_Var": metrics["Router_Norm_Var"],
                "Router_Norm_Min": metrics["Router_Norm_Min"],
                "Router_Norm_Max": metrics["Router_Norm_Max"],
                "Router_Norm_Mean": metrics["Router_Norm_Mean"]
            })
            
        # Создаем директорию под модель
        out_dir = os.path.join(SCRIPT_DIR, model_name)
        os.makedirs(out_dir, exist_ok=True)
        
        # Сохраняем все 5 файлов
        pd.DataFrame(m_router, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(out_dir, "metric_10_Router_Weights.csv"))
        pd.DataFrame({'SVD_Entropy_O_Proj': m_svd}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(out_dir, "metric_11_SVD_Entropy.csv"))
        pd.DataFrame({'Spectral_Norm': m_spec}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(out_dir, "metric_14_Spectral_Norm.csv"))
        pd.DataFrame({'Frobenius_Norm': m_frob}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(out_dir, "metric_15_Frobenius_Norm.csv"))
        pd.DataFrame({'Effective_Rank': m_erank}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(out_dir, "metric_16_Effective_Rank.csv"))
        
        print(f"Сохранено 5 файлов в {out_dir}/")
        
        del model
        if compute_device == "cuda":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
