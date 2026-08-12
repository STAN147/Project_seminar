import os
import re
import gc
import math
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy
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
            "ablations": os.path.join(SCRIPT_DIR, "ablations", task),
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

def get_logits_and_acc(model, tokenizer, prompt, target):
    messages = [{"role": "user", "content": prompt}]
    prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_formatted, return_tensors="pt").to(model.device)
    target_str = str(target).strip()
    target_ids = tokenizer(target_str, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    full_input_ids = torch.cat([inputs.input_ids, target_ids], dim=-1)
    target_len = target_ids.shape[1]
    with torch.no_grad():
        outputs = model(full_input_ids)
        logits = outputs.logits
        shift_logits = logits[..., -target_len-1:-1, :].contiguous()
        shift_labels = target_ids.contiguous()
        pred_tokens = torch.argmax(shift_logits, dim=-1)
        is_correct = torch.equal(pred_tokens, shift_labels)
    return logits, is_correct

class AblationHook:
    def __init__(self, layer):
        self.layer = layer
        self.hook = layer.register_forward_hook(self.ablate)
    def ablate(self, module, input_tuple, output_tuple):
        h_in = input_tuple[0]
        while isinstance(h_in, tuple):
            h_in = h_in[0]
        if isinstance(output_tuple, tuple):
            return (h_in,) + output_tuple[1:]
        return h_in
    def remove(self):
        self.hook.remove()

def compute_effective_rank(model, num_layers):
    m16_eff_ranks = []
    device = next(model.parameters()).device
    for l in range(num_layers):
        eff_ranks = []
        signature = f".{l}."
        for name, param in model.named_parameters():
            if signature in name and len(param.shape) == 2:
                W = param.detach().to(device).float()
                try:
                    S = torch.linalg.svdvals(W)
                    p = S / (S.sum() + 1e-9)
                    ent = -torch.sum(p * torch.log(p + 1e-9))
                    eff_ranks.append(torch.exp(ent).item())
                except:
                    pass
                del W
        m16_eff_ranks.append(np.mean(eff_ranks) if eff_ranks else 0.0)
    return m16_eff_ranks

def run_task_pipeline(model, tokenizer, task_name, samples):
    layers = get_layers(model)
    norm_layer = get_norm(model)
    lm_head = get_lm_head(model)
    num_layers = len(layers)
    baseline_correct = 0
    for sample in tqdm(samples, desc=f"  [Baseline {task_name.upper()}]"):
        _, is_correct = get_logits_and_acc(model, tokenizer, sample["prompt"], sample["target"])
        if is_correct:
            baseline_correct += 1
    baseline_acc = baseline_correct / len(samples)
    print(f"Baseline {task_name.upper()} Accuracy: {baseline_acc:.4f}")
    ablations_data = [{'Layer': 'Baseline', 'Accuracy': baseline_acc}]
    for i in tqdm(range(num_layers), desc=f"  [Ablations {task_name.upper()}]"):
        hook = AblationHook(layers[i])
        layer_correct = 0
        for sample in samples:
            _, is_correct = get_logits_and_acc(model, tokenizer, sample["prompt"], sample["target"])
            if is_correct:
                layer_correct += 1
        hook.remove()
        acc = layer_correct / len(samples)
        ablations_data.append({'Layer': i, 'Accuracy': acc})
    pd.DataFrame(ablations_data).to_csv(os.path.join(PATHS[task_name]['ablations'], "ablations.csv"), index=False)
    all_hiddens = {i: [] for i in range(num_layers + 1)}
    with torch.no_grad():
        for sample in tqdm(samples, desc=f"  [Скрытые состояния | {task_name.upper()}]"):
            messages = [{"role": "user", "content": sample["prompt"]}]
            prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_formatted, return_tensors="pt").to(model.device)
            outputs = model(**inputs, output_hidden_states=True)
            for i in range(num_layers + 1):
                h = outputs.hidden_states[i][0, -1, :].float().cpu().numpy()
                all_hiddens[i].append(h)
    avg_hiddens = {i: np.mean(all_hiddens[i], axis=0) for i in range(num_layers + 1)}
    m2_cos, m3_res, m5_l1 = [], [], []
    for i in range(num_layers):
        r_m2, r_m3, r_m5 = {'Layer': i}, {'Layer': i}, {'Layer': i}
        hi = avg_hiddens[i]
        for j in range(num_layers + 1):
            hj = avg_hiddens[j]
            r_m2[str(j)] = 1.0 - (np.dot(hi, hj) / (np.linalg.norm(hi) * np.linalg.norm(hj) + 1e-9))
            r_m3[str(j)] = np.linalg.norm(hj - hi) / (np.linalg.norm(hi) + 1e-9)
            r_m5[str(j)] = np.linalg.norm(hi - hj, ord=1)
        m2_cos.append(r_m2)
        m3_res.append(r_m3)
        m5_l1.append(r_m5)
    m_dir = PATHS[task_name]['metrics']
    pd.DataFrame(m2_cos).to_csv(os.path.join(m_dir, "metric_02_Cosine_Distance.csv"), index=False)
    pd.DataFrame(m3_res).to_csv(os.path.join(m_dir, "metric_03_Residual_Contribution.csv"), index=False)
    pd.DataFrame(m5_l1).to_csv(os.path.join(m_dir, "metric_05_L1_Distance.csv"), index=False)
    ll_accum = np.zeros(num_layers)
    for sample in tqdm(samples, desc=f"  [LogitLens | {task_name.upper()}]"):
        messages = [{"role": "user", "content": sample["prompt"]}]
        prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            base_logits = model(**inputs).logits[0, -1, :].cpu()
            base_argmax = torch.argmax(base_logits).item()
            outputs = model(**inputs, output_hidden_states=True)
            hiddens = outputs.hidden_states
        for i in range(num_layers):
            h_i = hiddens[i + 1][0, -1, :].to(model.device)
            h_norm = norm_layer(h_i)
            logits_i = lm_head(h_norm).cpu()
            prob_correct_token = F.softmax(logits_i.float(), dim=-1)[base_argmax].item()
            ll_accum[i] += prob_correct_token
    m13_ll = ll_accum / len(samples)
    pd.DataFrame({'LogitLens': m13_ll}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(m_dir, "metric_13_LogitLens.csv"))
    m17_var_shift = np.zeros(num_layers)
    for l in range(num_layers):
        h_matrix = torch.tensor(np.array(all_hiddens[l + 1])).float()
        token_vars = torch.var(h_matrix, dim=-1)
        m17_var_shift[l] = torch.std(token_vars).item()
    pd.DataFrame({'Variance_Shift': m17_var_shift}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(m_dir, "metric_17_Variance_shift.csv"))
    m18_outliers = np.zeros(num_layers)
    for l in range(num_layers):
        h_matrix = torch.tensor(np.array(all_hiddens[l + 1])).float()
        means = torch.mean(h_matrix, dim=0)
        stds = torch.std(h_matrix, dim=0) + 1e-9
        z_scores = torch.abs((h_matrix - means) / stds)
        outlier_mask = z_scores > 3.0
        m18_outliers[l] = torch.mean(outlier_mask.float()).item()
    pd.DataFrame({'Outlier_IoU': m18_outliers}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(m_dir, "metric_18_Outliner_driven_signals.csv"))

def main():
    print(f"Загрузка Llama-3.2-3B из {MODEL_DIR}...")
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
    m16_eff_rank = compute_effective_rank(model, num_layers)
    for task in TASKS:
        m_dir = PATHS[task]['metrics']
        pd.DataFrame({'Effective_Rank': m16_eff_rank}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(m_dir, "metric_16_Effective_Rank.csv"))
    for task in TASKS:
        samples = load_dataset(task)
        if samples:
            run_task_pipeline(model, tokenizer, task, samples)

if __name__ == "__main__":
    main()
