import os
import re
import math
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "copa_500.csv")

MODELS = ["gemma"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class AblationHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            return (input[0],) + output[1:]
        return input[0]
    def remove(self):
        self.hook.remove()

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

def evaluate_sample(model, tokenizer, question, choices, correct_idx):
    losses = []
    for choice in choices:
        text = question + " " + choice
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"], use_cache=False)
            losses.append(outputs.loss.item())
    pred_idx = np.argmin(losses)
    is_correct = int(pred_idx == correct_idx)
    target_loss = losses[correct_idx]
    return is_correct, target_loss

def compute_datafree_metrics(model, num_layers):
    m10, m11, m14, m15, m16 = [], [], [], [], []
    for l in range(num_layers):
        spec_norms, frob_norms, eff_ranks, svd_ents = [], [], [], []
        signature = f".{l}."
        for name, param in model.named_parameters():
            if signature in name and len(param.shape) == 2:
                W = param.detach().to(DEVICE).float()
                frob_norms.append(torch.norm(W, p='fro').item())
                try:
                    S = torch.linalg.svdvals(W)
                    spec_norms.append(S[0].item())
                    p = S / (S.sum() + 1e-9)
                    ent = -torch.sum(p * torch.log(p + 1e-9))
                    svd_ents.append(ent.item())
                    eff_ranks.append(torch.exp(ent).item())
                except:
                    pass
                del W
        m10.append(0.0) 
        m11.append(np.mean(svd_ents) if svd_ents else 0.0)
        m14.append(np.mean(spec_norms) if spec_norms else 0.0)
        m15.append(np.mean(frob_norms) if frob_norms else 0.0)
        m16.append(np.mean(eff_ranks) if eff_ranks else 0.0)
    return m10, m11, m14, m15, m16

def extract_hidden_geometries(model, tokenizer, df):
    num_layers = get_num_layers(model)
    all_hiddens = {l: [] for l in range(num_layers)}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  [Скрытые состояния]"):
        text = row['question'] + " " + row[row['answer']]
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            h_states = outputs.hidden_states[1:] 
            for l in range(num_layers):
                pooled = h_states[l].mean(dim=1).squeeze().cpu()
                all_hiddens[l].append(pooled)
    vectors = []
    for l in range(num_layers):
        vectors.append(torch.stack(all_hiddens[l]).mean(dim=0))
    V = torch.stack(vectors).float()
    m1, m2, m3, m4, m5, m6, m7, m8 = (np.zeros((num_layers, num_layers)) for _ in range(8))
    for i in range(num_layers):
        for j in range(num_layers):
            vi, vj = V[i], V[j]
            m1[i, j] = F.mse_loss(vi, vj).item()
            m2[i, j] = 1.0 - F.cosine_similarity(vi.unsqueeze(0), vj.unsqueeze(0)).item()
            m3[i, j] = torch.norm(vi - vj).item() / (torch.norm(vi).item() + 1e-9)
            m5[i, j] = torch.nn.functional.l1_loss(vi, vj).item()
            m6[i, j] = torch.max(torch.abs(vi - vj)).item()
            m7[i, j] = torch.var(vi).item() / (torch.var(vj).item() + 1e-9)
            vi_c = vi - vi.mean()
            vj_c = vj - vj.mean()
            m8[i, j] = (torch.dot(vi_c, vj_c) / (torch.norm(vi_c) * torch.norm(vj_c) + 1e-9)).item()
            m4[i, j] = m8[i, j] ** 2 
    return m1, m2, m3, m4, m5, m6, m7, m8

def compute_kl_and_logitlens(model, tokenizer, df):
    num_layers = get_num_layers(model)
    layers = get_layers(model)
    m12_kl = np.zeros(num_layers)
    m13_ll = np.zeros(num_layers)
    for l in tqdm(range(num_layers), desc="  [KL Noise & LogitLens]"):
        kl_batch, ll_batch = [], []
        for _, row in df.iterrows():
            text = row['question'] + " " + row[row['answer']]
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                base_out = model(**inputs, output_hidden_states=True, use_cache=False)
                base_logits = base_out.logits
                h = base_out.hidden_states[l+1]
                try:
                    if hasattr(model, 'lm_head'):
                        normed = model.model.norm(h) if hasattr(model.model, 'norm') else h
                        proj = model.lm_head(normed)
                        ent = -torch.sum(F.softmax(proj, dim=-1) * F.log_softmax(proj, dim=-1), dim=-1).mean().item()
                        ll_batch.append(ent)
                    else:
                        ll_batch.append(0.0)
                except:
                    ll_batch.append(0.0)

            def noise_hook(module, inp, out):
                if isinstance(out, tuple):
                    noise = torch.randn_like(out[0]) * 0.01
                    return (out[0] + noise,) + out[1:]
                return out + torch.randn_like(out) * 0.01

            hook = layers[l].register_forward_hook(noise_hook)
            with torch.no_grad():
                noise_out = model(**inputs, use_cache=False)
                noise_logits = noise_out.logits
            hook.remove()
            kl = F.kl_div(F.log_softmax(noise_logits, dim=-1), F.softmax(base_logits, dim=-1), reduction='batchmean').item()
            kl_batch.append(kl)
        m12_kl[l] = np.mean(kl_batch)
        m13_ll[l] = np.mean(ll_batch)
    return m12_kl, m13_ll

def main():
    df = pd.read_csv(DATASET_PATH)
    labels_map = {'A': 0, 'B': 1}
    for model_name in MODELS:
        model_path = os.path.join(MODELS_DIR, model_name)
        dir_abl = os.path.join(SCRIPT_DIR, "ablations", model_name)
        dir_met = os.path.join(SCRIPT_DIR, "metrics", model_name)
        dir_dfree = os.path.join(dir_met, "data-free")
        for d in [dir_abl, dir_met, dir_dfree]:
            os.makedirs(d, exist_ok=True)
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
        base_accs = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc="  [Baseline]"):
            correct_idx = labels_map[row['answer']]
            choices = [row['A'], row['B']]
            acc, _ = evaluate_sample(model, tokenizer, row['question'], choices, correct_idx)
            base_accs.append(acc)
        baseline_acc = np.mean(base_accs)
        ablation_results = []
        for l_idx in tqdm(range(num_layers), desc="  [Ablations]"):
            hook = AblationHook(get_layers(model)[l_idx])
            accs = []
            for i, row in df.iterrows():
                correct_idx = labels_map[row['answer']]
                choices = [row['A'], row['B']]
                acc, _ = evaluate_sample(model, tokenizer, row['question'], choices, correct_idx)
                accs.append(acc)
            hook.remove()
            abl_acc = np.mean(accs)
            ablation_results.append({'Layer': l_idx, 'Accuracy': abl_acc})
        pd.DataFrame([{'Layer': 'Baseline', 'Accuracy': baseline_acc}] + ablation_results).to_csv(os.path.join(dir_abl, "ablations.csv"), index=False)
        m1, m2, m3, m4, m5, m6, m7, m8 = extract_hidden_geometries(model, tokenizer, df)
        metrics_matrices = [
            (m1, "metric_01_MSE.csv"), (m2, "metric_02_Cosine_Distance.csv"),
            (m3, "metric_03_Residual_Contribution.csv"), (m4, "metric_04_CKA.csv"),
            (m5, "metric_05_L1_Distance.csv"), (m6, "metric_06_L_Infinity.csv"),
            (m7, "metric_07_Variance_Ratio.csv"), (m8, "metric_08_Pearson_Correlation.csv")
        ]
        for mat, fname in metrics_matrices:
            pd.DataFrame(mat, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(dir_met, fname))
        m12, m13 = compute_kl_and_logitlens(model, tokenizer, df)
        pd.DataFrame({'KL_noise': m12}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(dir_met, "metric_12_KL_noise.csv"))
        pd.DataFrame({'LogitLens': m13}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(dir_met, "metric_13_LogitLens.csv"))
        m10, m11, m14, m15, m16 = compute_datafree_metrics(model, num_layers)
        pd.DataFrame({'Router_Weights': m10}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(dir_dfree, "metric_10_Router_Weights.csv"))
        pd.DataFrame({'SVD_Entropy': m11}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(dir_dfree, "metric_11_SVD_Entropy.csv"))
        pd.DataFrame({'Spectral_Norm': m14}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(dir_dfree, "metric_14_Spectral_Norm.csv"))
        pd.DataFrame({'Frobenius_Norm': m15}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(dir_dfree, "metric_15_Frobenius_Norm.csv"))
        pd.DataFrame({'Effective_Rank': m16}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(dir_dfree, "metric_16_Effective_Rank.csv"))
        del model, tokenizer
        if DEVICE == "cuda": torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
