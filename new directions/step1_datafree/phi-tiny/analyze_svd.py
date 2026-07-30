import os
import json
import torch
import pandas as pd
from safetensors import safe_open

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

PHI_MODEL_DIR = os.path.join(BASE_DIR, "models", "phi-tiny")
SVD_DIR = os.path.join(SCRIPT_DIR, "svd_analysis")

os.makedirs(SVD_DIR, exist_ok=True)

def compute_svd_entropy(weight_tensor):
    W = weight_tensor.float()
    S = torch.linalg.svdvals(W)
    S_norm = S / torch.sum(S)
    entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-9))
    max_entropy = torch.log(torch.tensor(len(S), dtype=torch.float32))
    return (entropy / max_entropy).item()

def analyze_phi_svd():    
    index_path = os.path.join(PHI_MODEL_DIR, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)["weight_map"]
    num_layers = 32
    metrics = []
    for i in range(num_layers):
        possible_keys = [k for k in index_data.keys() if f".{i}." in k and ("o_proj" in k or "out_proj" in k or "dense" in k) and ("attn" in k or "mixer" in k) and "weight" in k]
        tensor_name = possible_keys[0]
        file_name = index_data[tensor_name]
        file_path = os.path.join(PHI_MODEL_DIR, file_name)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            weight = f.get_tensor(tensor_name)
        entropy = compute_svd_entropy(weight)
        metrics.append({
            "Layer": i,
            "SVD_Entropy_O_Proj": entropy
        })
    df = pd.DataFrame(metrics)
    df.set_index("Layer", inplace=True)
    out_path = os.path.join(SVD_DIR, "metric_11_SVD_Entropy_phi.csv")
    df.to_csv(out_path)

analyze_phi_svd()
