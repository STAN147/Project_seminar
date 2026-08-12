import os
import re
import gc
import math
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, entropy
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

MODEL_DIR = os.path.join(BASE_DIR, "models", "llama")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

TASKS = ["csqa", "siqa"]

def setup_directories():
    paths = {}
    for task in TASKS:
        paths[task] = {
            "metrics": os.path.join(SCRIPT_DIR, "metrics", task)
        }
        for path in paths[task].values():
            os.makedirs(path, exist_ok=True)
    return paths

PATHS = setup_directories()

def get_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    if hasattr(model, 'layers'):
        return model.layers
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 5:
            if 'vision' not in name.lower():
                return module
    raise AttributeError(f"Не удалось определить слои для модели {type(model)}")

def get_norm(model):
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        return model.model.norm
    if hasattr(model, 'norm'):
        return model.norm
    for name, module in model.named_modules():
        if 'norm' in name.lower() and not isinstance(module, nn.ModuleList):
            return module
    return nn.Identity()

def get_lm_head(model):
    if hasattr(model, 'lm_head'):
        return model.lm_head
    for name, module in model.named_modules():
        if ("lm_head" in name or "head" in name) and isinstance(module, nn.Linear):
            if "vision" not in name.lower():
                return module
    raise AttributeError("Не удалось найти lm_head")

def load_dataset(task):
    samples = []
    if task == "csqa":
        path = os.path.join(DATASETS_DIR, "csqa.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                raw_prompt = str(row['prompt']).replace("\nAnswer:", "")
                instruct_prompt = f"{raw_prompt}\n\nInstruction: Output ONLY the single correct letter (A, B, C, D, or E). Do not write any explanations, punctuation, or spaces.\nAnswer:"
                samples.append({"prompt": instruct_prompt, "target": str(row['target']).strip()})       
    elif task == "siqa":
        path = os.path.join(DATASETS_DIR, "siqa_500.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                context = str(row.get('context', ''))
                question = str(row.get('question', ''))
                ans1 = str(row.get('answerA', row.get('choice1', '')))
                ans2 = str(row.get('answerB', row.get('choice2', '')))
                ans3 = str(row.get('answerC', row.get('choice3', '')))
                instruct_prompt = (
                    f"Context: {context}\n"
                    f"Question: {question}\n"
                    f"1) {ans1}\n"
                    f"2) {ans2}\n"
                    f"3) {ans3}\n\n"
                    f"Instruction: Output ONLY the single correct digit (1, 2, or 3). Do not write any explanations, punctuation, or spaces.\nAnswer:"
                )
                samples.append({"prompt": instruct_prompt, "target": str(row.get('label', '1')).strip()})
    return samples

def compute_datafree_metrics(model, num_layers, compute_device):
    is_moe = any(("gate" in name or "router" in name) and "gate_proj" not in name for name, _ in model.named_parameters())
    m_spec, m_frob, m_erank = [], [], []
    m_svd, m_router = [], []
    for layer_idx in tqdm(range(num_layers), desc="[Data-Free] M10, M11, M14, M15, M16"):
        layer_signature = f".{layer_idx}."
        spectral_norms, frob_norms, eff_ranks = [], [], []
        router_metrics = {
            "Router_Norm_Var": 0.0,
            "Router_Norm_Min": 0.0,
            "Router_Norm_Max": 0.0,
            "Router_Norm_Mean": 0.0
        }
        svd_entropy_o_proj = 0.0
        router_found = False
        svd_found = False
        for name, param in model.named_parameters():
            if layer_signature in name and "weight" in name and len(param.shape) == 2:
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
                    if not svd_found and ("o_proj" in name or "out_proj" in name or "dense" in name) and ("attn" in name or "mixer" in name):
                        max_entropy = torch.log(torch.tensor(len(S), dtype=torch.float32))
                        svd_entropy_o_proj = (ent / max_entropy).item()
                        svd_found = True
                except Exception:
                    pass
                if is_moe and not router_found and ("gate" in name or "router" in name) and "gate_proj" not in name:
                    norms = torch.norm(W, p=2, dim=1)
                    if len(norms) > 1:
                        router_metrics["Router_Norm_Var"] = torch.var(norms).item()
                    router_metrics["Router_Norm_Min"] = torch.min(norms).item()
                    router_metrics["Router_Norm_Max"] = torch.max(norms).item()
                    router_metrics["Router_Norm_Mean"] = torch.mean(norms).item()
                    router_found = True
                del W
        m_spec.append(np.mean(spectral_norms) if spectral_norms else 0.0)
        m_frob.append(np.mean(frob_norms) if frob_norms else 0.0)
        m_erank.append(np.mean(eff_ranks) if eff_ranks else 0.0)
        m_svd.append(svd_entropy_o_proj)
        m_router.append(router_metrics)
    df_spec = pd.DataFrame({'Spectral_Norm': m_spec}, index=range(num_layers)).rename_axis('Layer')
    df_frob = pd.DataFrame({'Frobenius_Norm': m_frob}, index=range(num_layers)).rename_axis('Layer')
    df_svd = pd.DataFrame({'SVD_Entropy_O_Proj': m_svd}, index=range(num_layers)).rename_axis('Layer')
    df_router = pd.DataFrame(m_router, index=range(num_layers)).rename_axis('Layer')
    return df_spec, df_frob, df_svd, df_router

def run_task_missing_pipeline(model, tokenizer, task_name, samples):
    layers = get_layers(model)
    norm_layer = get_norm(model)
    lm_head = get_lm_head(model)
    num_layers = len(layers)
    m_dir = PATHS[task_name]['metrics']
    device = model.device
    all_hiddens = {i: [] for i in range(num_layers + 1)}
    with torch.no_grad():
        for sample in tqdm(samples, desc=f"  [Скрытые состояния | {task_name.upper()}]"):
            messages = [{"role": "user", "content": sample["prompt"]}]
            prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_formatted, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True)
            for i in range(num_layers + 1):
                h = outputs.hidden_states[i][0, -1, :].float().cpu().numpy()
                all_hiddens[i].append(h)
    avg_hiddens = {i: np.mean(all_hiddens[i], axis=0) for i in range(num_layers + 1)}
    m1_mse, m4_cka = [], []
    m6_linf, m7_var, m8_pear = [], [], []
    for i in range(num_layers):
        r_m1, r_m4 = {'Layer': i}, {'Layer': i}
        r_m6, r_m7, r_m8 = {'Layer': i}, {'Layer': i}, {'Layer': i}
        hi = avg_hiddens[i]
        for j in range(num_layers + 1):
            hj = avg_hiddens[j]
            r_m1[str(j)] = np.mean((hi - hj)**2)
            r_m4[str(j)] = pearsonr(hi, hj)[0] if np.std(hi) > 0 and np.std(hj) > 0 else 0
            r_m6[str(j)] = np.linalg.norm(hi - hj, ord=np.inf)
            r_m7[str(j)] = np.var(hj) / (np.var(hi) + 1e-9)
            r_m8[str(j)] = pearsonr(hi, hj)[0] if np.std(hi) > 0 and np.std(hj) > 0 else 0
        m1_mse.append(r_m1)
        m4_cka.append(r_m4)
        m6_linf.append(r_m6)
        m7_var.append(r_m7)
        m8_pear.append(r_m8)
    pd.DataFrame(m1_mse).to_csv(os.path.join(m_dir, "metric_01_MSE.csv"), index=False)
    pd.DataFrame(m4_cka).to_csv(os.path.join(m_dir, "metric_04_CKA.csv"), index=False)
    pd.DataFrame(m6_linf).to_csv(os.path.join(m_dir, "metric_06_L_Infinity.csv"), index=False)
    pd.DataFrame(m7_var).to_csv(os.path.join(m_dir, "metric_07_Variance_Ratio.csv"), index=False)
    pd.DataFrame(m8_pear).to_csv(os.path.join(m_dir, "metric_08_Pearson_Correlation.csv"), index=False)
    num_eval_samples = 20
    eval_samples = samples[:num_eval_samples]
    kl_accum = np.zeros(num_layers)
    for sample in tqdm(eval_samples, desc=f"  [M12 KL Noise | {task_name.upper()}]"):
        messages = [{"role": "user", "content": sample["prompt"]}]
        prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_formatted, return_tensors="pt").to(device)
        with torch.no_grad():
            base_logits = model(**inputs).logits[0, -1, :].cpu()
            base_probs = F.softmax(base_logits.float(), dim=-1).numpy()
            outputs = model(**inputs, output_hidden_states=True)
            hiddens = outputs.hidden_states
        for i in range(num_layers):
            h_i = hiddens[i][0, -1, :].to(device) 
            noise = torch.randn_like(h_i) * 0.1
            h_noisy = norm_layer(h_i + noise)
            logits_noisy = lm_head(h_noisy).cpu()
            noisy_probs = F.softmax(logits_noisy.float(), dim=-1).detach().numpy()
            kl_val = entropy(base_probs, noisy_probs + 1e-9)
            kl_accum[i] += kl_val
    kl_data = [{'Layer': i, 'Value': kl_accum[i] / num_eval_samples} for i in range(num_layers)]
    pd.DataFrame(kl_data).to_csv(os.path.join(m_dir, "metric_12_KL_noise.csv"), index=False)
    dummy_moe = pd.DataFrame([{'Layer': i, 'Value': 0.0} for i in range(num_layers)])
    dummy_moe.to_csv(os.path.join(m_dir, "metric_09_Router_Entropy.csv"), index=False)
    step5_samples = samples[:500]
    metrics_data = {
        'var_shift': {i: [] for i in range(num_layers)},
        'outlier_iou': {i: [] for i in range(num_layers)},
        'attn_norm': {i: [] for i in range(num_layers)},
        'mlp_norm': {i: [] for i in range(num_layers)}
    }

    def get_layer_hook(layer_idx):
        def hook(module, inp, output):
            x_in = inp[0].float()
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
        hooks.append(layer.register_forward_hook(get_layer_hook(i)))
        if hasattr(layer, 'self_attn'):
            hooks.append(layer.self_attn.register_forward_hook(get_attn_hook(i)))
        if hasattr(layer, 'mlp'):
            hooks.append(layer.mlp.register_forward_hook(get_mlp_hook(i)))
    for sample in tqdm(step5_samples, desc=f"  [Step5 Инференс | {task_name.upper()}]"):
        messages = [{"role": "user", "content": sample["prompt"]}]
        prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_formatted, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            _ = model(**inputs, use_cache=False)
    for h in hooks:
        h.remove()
    res_var_shift, res_outlier, res_ratio = [], [], []
    for i in range(num_layers):
        avg_var_shift = np.mean(metrics_data['var_shift'][i]) if metrics_data['var_shift'][i] else 0.0
        avg_outlier = np.mean(metrics_data['outlier_iou'][i]) if metrics_data['outlier_iou'][i] else 0.0
        avg_attn = np.mean(metrics_data['attn_norm'][i]) if metrics_data['attn_norm'][i] else 0.0
        avg_mlp = np.mean(metrics_data['mlp_norm'][i]) if metrics_data['mlp_norm'][i] else 0.0
        ratio = (avg_attn / (avg_mlp + 1e-6)) if avg_mlp > 0 else 0.0
        res_var_shift.append({'Layer': i, 'Variance_Shift': avg_var_shift})
        res_outlier.append({'Layer': i, 'Outlier_IoU': avg_outlier})
        res_ratio.append({'Layer': i, 'Attn_vs_MLP_Ratio': ratio})
    pd.DataFrame(res_var_shift).to_csv(os.path.join(m_dir, "metric_17_Variance_shift.csv"), index=False)
    pd.DataFrame(res_outlier).to_csv(os.path.join(m_dir, "metric_18_Outliner_driven_signals.csv"), index=False)
    pd.DataFrame(res_ratio).to_csv(os.path.join(m_dir, "metric_19_Att_vs_MLP_ratio.csv"), index=False)

def main():
    compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        local_files_only=True,
        trust_remote_code=True
    ).eval()
    num_layers = len(get_layers(model))
    df_spec, df_frob, df_svd, df_router = compute_datafree_metrics(model, num_layers, compute_device)
    for task in TASKS:
        t_dir = PATHS[task]['metrics']
        df_spec.to_csv(os.path.join(t_dir, "metric_14_Spectral_Norm.csv"))
        df_frob.to_csv(os.path.join(t_dir, "metric_15_Frobenius_Norm.csv"))
        df_svd.to_csv(os.path.join(t_dir, "metric_11_SVD_Entropy.csv"))
        df_router.to_csv(os.path.join(t_dir, "metric_10_Router_Weights.csv"))
    for task in TASKS:
        samples = load_dataset(task)
        if samples:
            run_task_missing_pipeline(model, tokenizer, task, samples)

if __name__ == "__main__":
    main()
