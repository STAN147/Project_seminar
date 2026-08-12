import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../models"))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../datasets"))

MODELS = ["gemma"]
TASKS = ["csqa", "siqa"]

METRICS = {
    "Variance shift": "metric_17_Variance_shift.csv",
    "Outliner-driven signals": "metric_18_Outliner_driven_signals.csv",
    "Attention vs MLP Contribution Ratio": "metric_19_Att_vs_MLP_ratio.csv"
}

for metric_dir in METRICS.keys():
    for model in MODELS:
        for task in TASKS:
            os.makedirs(os.path.join(BASE_DIR, metric_dir, model, task), exist_ok=True)

def get_layer_name(idx, model_name):
    return f"MoE_Layer_{idx}" if model_name in ["Qwen", "phi-tiny"] else f"Layer_{idx}"

def get_layers(model):
    max_len = 0
    best_layers = None
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > max_len:
            max_len = len(module)
            best_layers = module
    if best_layers is not None:
        return best_layers
    raise ValueError("Could not find layers in the model.")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for model_name in MODELS:
        print(f"\nRunning {model_name}")
        model_path = os.path.join(MODELS_DIR, model_name)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if model_name == "Qwen":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map={"": 0},
                    quantization_config=quantization_config,
                    trust_remote_code=True
                )
            elif model_name == "gemma":
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    device_map={"": 0}, 
                    dtype=torch.float16,
                    trust_remote_code=True
                )
            model.eval()
        except Exception as e:
            print(f"Ошибка загрузки модели {model_name}: {e}")
            continue
        try:
            layers = get_layers(model)
        except Exception as e:
            print(f"Ошибка поиска слоев для {model_name}: {e}")
            continue
        num_layers = len(layers)
        for task in TASKS:
            print(f"[{model_name}] | {task.upper()}")
            dataset_path = os.path.join(DATA_DIR, f"{task}.csv" if task == "csqa" else f"{task}_500.csv")
            try:
                df = pd.read_csv(dataset_path)
                text_col = 'text' if 'text' in df.columns else df.columns[0]
                texts = df[text_col].dropna().astype(str).tolist()[:500] 
            except Exception as e:
                print(f"Ошибка чтения {dataset_path}: {e}")
                continue
            metrics_data = {
                'var_shift': {i: [] for i in range(num_layers)},
                'outlier_iou': {i: [] for i in range(num_layers)},
                'attn_norm': {i: [] for i in range(num_layers)},
                'mlp_norm': {i: [] for i in range(num_layers)}
            }

            def get_layer_hook(layer_idx):
                def hook(module, input, output):
                    x_in = input[0].float()
                    x_out = output[0].float() if isinstance(output, tuple) else output.float()
                    var_in = x_in.var(dim=-1)
                    var_out = x_out.var(dim=-1)
                    shift = (var_out / (var_in + 1e-6)).mean().item()
                    metrics_data['var_shift'][layer_idx].append(shift)
                    k = max(1, int(x_in.shape[-1] * 0.01))
                    _, top_in_idx = torch.topk(x_in.abs(), k, dim=-1)
                    _, top_out_idx = torch.topk(x_out.abs(), k, dim=-1)
                    intersection = (top_in_idx.unsqueeze(-1) == top_out_idx.unsqueeze(-2)).any(dim=-1).sum(dim=-1).float()
                    union = (2 * k) - intersection
                    iou = (intersection / (union + 1e-6)).mean().item()
                    metrics_data['outlier_iou'][layer_idx].append(iou)
                return hook

            def get_attn_hook(layer_idx):
                def hook(module, input, output):
                    out = output[0].float() if isinstance(output, tuple) else output.float()
                    metrics_data['attn_norm'][layer_idx].append(out.norm(dim=-1).mean().item())
                return hook

            def get_mlp_hook(layer_idx):
                def hook(module, input, output):
                    out = output[0].float() if isinstance(output, tuple) else output.float()
                    metrics_data['mlp_norm'][layer_idx].append(out.norm(dim=-1).mean().item())
                return hook

            hooks = []
            for i, layer in enumerate(layers):
                hooks.append(layer.register_forward_hook(get_layer_hook(i)))
                if hasattr(layer, 'self_attn'):
                    hooks.append(layer.self_attn.register_forward_hook(get_attn_hook(i)))
                if hasattr(layer, 'mlp'):
                    hooks.append(layer.mlp.register_forward_hook(get_mlp_hook(i)))
            for text in tqdm(texts, desc=f"Inference {model_name} - {task}"):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                with torch.no_grad():
                    _ = model(**inputs, use_cache=False)
            for h in hooks:
                h.remove()
            res_var_shift = []
            res_outlier = []
            res_ratio = []
            for i in range(num_layers):
                layer_id = get_layer_name(i, model_name)
                avg_var_shift = np.mean(metrics_data['var_shift'][i]) if metrics_data['var_shift'][i] else 0.0
                avg_outlier = np.mean(metrics_data['outlier_iou'][i]) if metrics_data['outlier_iou'][i] else 0.0
                avg_attn = np.mean(metrics_data['attn_norm'][i]) if metrics_data['attn_norm'][i] else 0.0
                avg_mlp = np.mean(metrics_data['mlp_norm'][i]) if metrics_data['mlp_norm'][i] else 0.0
                ratio = (avg_attn / (avg_mlp + 1e-6)) if avg_mlp > 0 else 0.0
                res_var_shift.append((layer_id, avg_var_shift))
                res_outlier.append((layer_id, avg_outlier))
                res_ratio.append((layer_id, ratio))
            df_vs = pd.DataFrame(res_var_shift, columns=["Layer", "Variance_Shift"])
            path_vs = os.path.join(BASE_DIR, "Variance shift", model_name, task, METRICS["Variance shift"])
            df_vs.to_csv(path_vs, index=False)
            df_out = pd.DataFrame(res_outlier, columns=["Layer", "Outlier_IoU"])
            path_out = os.path.join(BASE_DIR, "Outliner-driven signals", model_name, task, METRICS["Outliner-driven signals"])
            df_out.to_csv(path_out, index=False)
            df_ratio = pd.DataFrame(res_ratio, columns=["Layer", "Attn_vs_MLP_Ratio"])
            path_ratio = os.path.join(BASE_DIR, "Attention vs MLP Contribution Ratio", model_name, task, METRICS["Attention vs MLP Contribution Ratio"])
            df_ratio.to_csv(path_ratio, index=False)
        del model
        del tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
