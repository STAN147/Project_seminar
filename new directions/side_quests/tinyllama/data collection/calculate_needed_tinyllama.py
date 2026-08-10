import os
import re
import math
import torch
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models", "tinyllama")
CSQA_PATH = os.path.join(BASE_DIR, "datasets", "csqa.csv")
SIQA_PATH = os.path.join(BASE_DIR, "datasets", "siqa_500.csv")

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
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    if hasattr(model, 'layers'):
        return model.layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 5:
            if 'vision' not in name.lower():
                return module

def get_num_layers(model):
    return len(get_layers(model))

def evaluate_csqa_sample(model, tokenizer, question, choices, correct_idx):
    losses = []
    for choice in choices:
        text = f"{question} {choice}".strip()
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"], use_cache=False)
            losses.append(outputs.loss.item())
    pred_idx = np.argmin(losses)
    is_correct = int(pred_idx == correct_idx)
    target_loss = losses[correct_idx]
    return is_correct, target_loss

def evaluate_siqa_sample(model, tokenizer, context, question, answers, correct_idx):
    losses = []
    for ans in answers:
        text = f"{context} {question} {ans}".strip()
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"], use_cache=False)
            losses.append(outputs.loss.item())
    pred_idx = np.argmin(losses)
    is_correct = int(pred_idx == correct_idx)
    target_loss = losses[correct_idx]
    return is_correct, target_loss

def parse_csqa_prompt(prompt, target):
    parts = prompt.split('\nChoices: ')
    q = parts[0].replace('Question: ', '').strip()
    choices_part = parts[1].split('\nAnswer:')[0].strip()
    pattern = r'([A-E])\)\s*(.*?)(?=\s*[A-E]\)|$)'
    matches = re.findall(pattern, choices_part)
    labels = [m[0] for m in matches]
    choices = [m[1].strip() for m in matches]
    target_idx = labels.index(target.strip()) if target.strip() in labels else 0
    return q, choices, target_idx

def compute_datafree_metrics(model, num_layers):
    m16_eff_ranks = []
    for l in range(num_layers):
        eff_ranks = []
        signature = f".{l}."
        for name, param in model.named_parameters():
            if signature in name and len(param.shape) == 2:
                W = param.detach().to(DEVICE).float()
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

def extract_metrics_for_task(model, tokenizer, texts, task_name):
    num_layers = get_num_layers(model)
    layers = get_layers(model)
    
    all_hiddens = {l: [] for l in range(num_layers)}
    for text in tqdm(texts, desc=f"  [Скрытые состояния ({len(texts)}) | {task_name}]"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            h_states = outputs.hidden_states[1:]
            for l in range(num_layers):
                pooled = h_states[l].mean(dim=1).squeeze().cpu()
                all_hiddens[l].append(pooled)
                
    vectors = [torch.stack(all_hiddens[l]).mean(dim=0) for l in range(num_layers)]
    V = torch.stack(vectors).float()
    
    m2_cosine = np.zeros((num_layers, num_layers))
    m3_residual = np.zeros((num_layers, num_layers))
    m5_l1 = np.zeros((num_layers, num_layers))
    
    for i in range(num_layers):
        for j in range(num_layers):
            vi, vj = V[i], V[j]
            m2_cosine[i, j] = 1.0 - F.cosine_similarity(vi.unsqueeze(0), vj.unsqueeze(0)).item()
            m3_residual[i, j] = torch.norm(vi - vj).item() / (torch.norm(vi).item() + 1e-9)
            m5_l1[i, j] = F.l1_loss(vi, vj).item()
            
    m13_ll = np.zeros(num_layers)
    for l in tqdm(range(num_layers), desc=f"  [LogitLens ({len(texts)}) | {task_name}]"):
        ll_batch = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
            with torch.no_grad():
                base_out = model(**inputs, output_hidden_states=True, use_cache=False)
                h = base_out.hidden_states[l+1]
                try:
                    normed = model.model.norm(h) if hasattr(model.model, 'norm') else h
                    proj = model.lm_head(normed)
                    ent = -torch.sum(F.softmax(proj, dim=-1) * F.log_softmax(proj, dim=-1), dim=-1).mean().item()
                    ll_batch.append(ent)
                except:
                    ll_batch.append(0.0)
        m13_ll[l] = np.mean(ll_batch)
        
    m17_var_shift = np.zeros(num_layers)
    for l in range(num_layers):
        h_matrix = torch.stack(all_hiddens[l]).float()
        token_vars = torch.var(h_matrix, dim=-1)
        m17_var_shift[l] = torch.std(token_vars).item()
        
    m18_outliers = np.zeros(num_layers)
    for l in range(num_layers):
        h_matrix = torch.stack(all_hiddens[l]).float()
        means = torch.mean(h_matrix, dim=0)
        stds = torch.std(h_matrix, dim=0) + 1e-9
        z_scores = torch.abs((h_matrix - means) / stds)
        outlier_mask = z_scores > 3.0
        m18_outliers[l] = torch.mean(outlier_mask.float()).item()

    return m2_cosine, m3_residual, m5_l1, m13_ll, m17_var_shift, m18_outliers

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True
    ).eval()
    model.config.use_cache = False
    
    num_layers = get_num_layers(model)
    layers = get_layers(model)
    
    m16_eff_rank = compute_datafree_metrics(model, num_layers)

    tasks = ['csqa', 'siqa']
    for task in tasks:
        dir_abl = os.path.join(SCRIPT_DIR, "ablations", task)
        dir_met = os.path.join(SCRIPT_DIR, "metrics", task)
        os.makedirs(dir_abl, exist_ok=True)
        os.makedirs(dir_met, exist_ok=True)
        
        texts_for_features = []
        
        if task == 'csqa':
            df_csqa = pd.read_csv(CSQA_PATH)
            samples = []
            for _, row in df_csqa.iterrows():
                q, choices, target_idx = parse_csqa_prompt(row['prompt'], row['target'])
                samples.append((q, choices, target_idx))
                texts_for_features.append(f"{q} {choices[target_idx]}")
                    
            print(f"\n[TinyLlama | CSQA] Полная выборка: {len(samples)} вопросов. Расчет Baseline...")
            base_accs = [evaluate_csqa_sample(model, tokenizer, q, ch, idx)[0] for q, ch, idx in tqdm(samples, desc=f"  [Baseline CSQA ({len(samples)})]")]
            baseline_acc = np.mean(base_accs)
            print(f"Baseline CSQA Accuracy: {baseline_acc:.4f}")
            
            abl_res = []
            for l in tqdm(range(num_layers), desc=f"  [Ablations CSQA ({len(samples)})]"):
                hook = AblationHook(layers[l])
                accs = [evaluate_csqa_sample(model, tokenizer, q, ch, idx)[0] for q, ch, idx in samples]
                hook.remove()
                abl_res.append({'Layer': l, 'Accuracy': np.mean(accs)})
                
            pd.DataFrame([{'Layer': 'Baseline', 'Accuracy': baseline_acc}] + abl_res).to_csv(os.path.join(dir_abl, "ablations.csv"), index=False)

        elif task == 'siqa':
            df_siqa = pd.read_csv(SIQA_PATH)
            samples = []
            for _, row in df_siqa.iterrows():
                ctx = str(row['context']) if pd.notna(row.get('context')) else ""
                q = str(row['question'])
                answers = [str(row['answerA']), str(row['answerB']), str(row['answerC'])]
                correct_idx = int(row['label']) - 1 if str(row['label']).isdigit() else 0
                samples.append((ctx, q, answers, correct_idx))
                texts_for_features.append(f"{ctx} {q} {answers[correct_idx]}".strip())
                
            print(f"\n[TinyLlama | SIQA] Полная выборка: {len(samples)} вопросов. Расчет Baseline...")
            base_accs = [evaluate_siqa_sample(model, tokenizer, c, q, a, idx)[0] for c, q, a, idx in tqdm(samples, desc=f"  [Baseline SIQA ({len(samples)})]")]
            baseline_acc = np.mean(base_accs)
            print(f"Baseline SIQA Accuracy: {baseline_acc:.4f}")
            
            abl_res = []
            for l in tqdm(range(num_layers), desc=f"  [Ablations SIQA ({len(samples)})]"):
                hook = AblationHook(layers[l])
                accs = [evaluate_siqa_sample(model, tokenizer, c, q, a, idx)[0] for c, q, a, idx in samples]
                hook.remove()
                abl_res.append({'Layer': l, 'Accuracy': np.mean(accs)})
                
            pd.DataFrame([{'Layer': 'Baseline', 'Accuracy': baseline_acc}] + abl_res).to_csv(os.path.join(dir_abl, "ablations.csv"), index=False)

        m2, m3, m5, m13, m17, m18 = extract_metrics_for_task(model, tokenizer, texts_for_features, task)
        
        pd.DataFrame(m2, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(dir_met, "metric_02_Cosine_Distance.csv"))
        pd.DataFrame(m3, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(dir_met, "metric_03_Residual_Contribution.csv"))
        pd.DataFrame(m5, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(dir_met, "metric_05_L1_Distance.csv"))
        pd.DataFrame({'LogitLens': m13}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(dir_met, "metric_13_LogitLens.csv"))
        pd.DataFrame({'Effective_Rank': m16_eff_rank}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(dir_met, "metric_16_Effective_Rank.csv"))
        pd.DataFrame({'Variance_Shift': m17}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(dir_met, "metric_17_Variance_shift.csv"))
        pd.DataFrame({'Outlier_IoU': m18}, index=range(num_layers)).rename_axis('Layer').to_csv(os.path.join(dir_met, "metric_18_Outliner_driven_signals.csv"))

if __name__ == "__main__":
    main()
