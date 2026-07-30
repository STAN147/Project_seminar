import os
import json
import torch
import pandas as pd
from safetensors import safe_open

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

PHI_MODEL_DIR = os.path.join(BASE_DIR, "models", "phi-tiny")
ROUTER_DIR = os.path.join(SCRIPT_DIR, "router analysis")

os.makedirs(ROUTER_DIR, exist_ok=True)

def analyze_phi_routers():
    index_path = os.path.join(PHI_MODEL_DIR, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)["weight_map"]
    num_layers = 32
    metrics = []
    for i in range(num_layers):
        possible_keys = [k for k in index_data.keys() if f".{i}." in k and ("gate" in k or "router" in k) and "weight" in k]            
        tensor_name = possible_keys[0]
        file_name = index_data[tensor_name]
        file_path = os.path.join(PHI_MODEL_DIR, file_name)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            weight = f.get_tensor(tensor_name)
        norms = torch.norm(weight.float(), p=2, dim=1)
        metrics.append({
            "Layer": i,
            "Router_Norm_Var": torch.var(norms).item(),
            "Router_Norm_Min": torch.min(norms).item(),
            "Router_Norm_Max": torch.max(norms).item(),
            "Router_Norm_Mean": torch.mean(norms).item()
        })
    df = pd.DataFrame(metrics)
    df.set_index("Layer", inplace=True)    
    out_path = os.path.join(ROUTER_DIR, "metric_10_Router_Weights.csv")
    df.to_csv(out_path)

analyze_phi_routers()
