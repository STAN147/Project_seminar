import os
import json
import torch
import pandas as pd
from safetensors import safe_open

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
QWEN_MODEL_DIR = os.path.join(BASE_DIR, "models", "Qwen")
ROUTER_DIR = os.path.join(SCRIPT_DIR, "router analysis")

os.makedirs(ROUTER_DIR, exist_ok=True)
def analyze_qwen_routers():
    index_path = os.path.join(QWEN_MODEL_DIR, "model.safetensors.index.json")        
    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)["weight_map"]
    num_layers = 24
    metrics = []
    for i in range(num_layers):
        tensor_name = f"model.layers.{i}.mlp.gate.weight"
        file_name = index_data[tensor_name]
        file_path = os.path.join(QWEN_MODEL_DIR, file_name)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            weight = f.get_tensor(tensor_name)
        norms = torch.norm(weight.float(), p=2, dim=1)
        var_norm = torch.var(norms).item()
        min_norm = torch.min(norms).item()
        max_norm = torch.max(norms).item()
        mean_norm = torch.mean(norms).item()
        metrics.append({
            "Layer": i,
            "Router_Norm_Var": var_norm,
            "Router_Norm_Min": min_norm,
            "Router_Norm_Max": max_norm,
            "Router_Norm_Mean": mean_norm
        })
    df = pd.DataFrame(metrics)
    df.set_index("Layer", inplace=True)
    out_path = os.path.join(ROUTER_DIR, "metric_10_Router_Weights.csv")
    df.to_csv(out_path)

analyze_qwen_routers()
