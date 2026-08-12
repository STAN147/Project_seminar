import os
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr, entropy
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models", "gemma")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

TASKS = ["csqa"]
MAX_SAMPLES = 5000

def setup_directories():
    paths = {}
    for task in TASKS:
        task_dir = os.path.join(SCRIPT_DIR, task)
        paths[task] = {
            "ablations": os.path.join(task_dir, "ablations"),
            "metrics": os.path.join(task_dir, "metrics"),
            "data-free": os.path.join(task_dir, "data-free")
        }
        for path in paths[task].values():
            os.makedirs(path, exist_ok=True)
    return paths

PATHS = setup_directories()

def get_layers(model):
    max_len = 0
    best_layers = None
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > max_len:
            max_len = len(module)
            best_layers = module
    if best_layers is not None:
        return best_layers

def get_norm(model):
    max_len = 0
    best_parent = None
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.ModuleList) and len(child) > max_len:
                max_len = len(child)
                best_parent = module
    if best_parent is not None:
        for child_name, child in best_parent.named_children():
            if "norm" in child_name.lower():
                return child
    return nn.Identity()

def get_lm_head(model):
    for name, module in model.named_modules():
        if ("lm_head" in name or "head" in name) and isinstance(module, nn.Linear):
            if "vision" not in name.lower():
                return module
    return model.lm_head

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
    return samples[:MAX_SAMPLES]

def get_logits_and_acc(model, tokenizer, prompt, target):
    messages = [{"role": "user", "content": prompt}]
    prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_formatted, return_tensors="pt").to(model.device)
    target = str(target).strip()
    target_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
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

def calculate_svd_entropy(model, num_layers, paths):
    svd_entropies = []
    layers = get_layers(model)
    for i in tqdm(range(num_layers), desc="SVD"):
        weight = layers[i].mlp.down_proj.weight.detach().float()
        _, S, _ = torch.svd(weight)
        S_norm = S / S.sum()
        S_norm = S_norm[S_norm > 0]
        entropy_val = -torch.sum(S_norm * torch.log(S_norm)).item()
        svd_entropies.append({'Layer': i, 'Value': entropy_val})
    df_svd = pd.DataFrame(svd_entropies)
    for task in TASKS:
        df_svd.to_csv(os.path.join(PATHS[task]['data-free'], "metric_11_SVD_Entropy.csv"), index=False)

def run_task_pipeline(model, tokenizer, task_name, samples):
    layers = get_layers(model)
    norm_layer = get_norm(model)
    lm_head = get_lm_head(model)
    num_layers = len(layers)
    baseline_correct = 0
    for sample in tqdm(samples, desc="Baseline"):
        _, is_correct = get_logits_and_acc(model, tokenizer, sample["prompt"], sample["target"])
        if is_correct:
            baseline_correct += 1
    baseline_acc = baseline_correct / len(samples)
    ablations_data = [{'Layer': 'Baseline', 'Accuracy': baseline_acc}]
    for i in tqdm(range(num_layers), desc="Ablating Layers"):
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
        for sample in tqdm(samples, desc="Extracting Hiddens"):
            messages = [{"role": "user", "content": sample["prompt"]}]
            prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_formatted, return_tensors="pt").to(model.device)
            outputs = model(**inputs, output_hidden_states=True)
            for i in range(num_layers + 1):
                h = outputs.hidden_states[i][0, -1, :].float().cpu().numpy()
                all_hiddens[i].append(h)
    avg_hiddens = {i: np.mean(all_hiddens[i], axis=0) for i in range(num_layers + 1)}
    m1_mse, m2_cos, m3_res, m4_cka = [], [], [], []
    m5_l1, m6_linf, m7_var, m8_pear = [], [], [], []
    for i in range(num_layers):
        r_m1, r_m2, r_m3, r_m4 = {'Layer': i}, {'Layer': i}, {'Layer': i}, {'Layer': i}
        r_m5, r_m6, r_m7, r_m8 = {'Layer': i}, {'Layer': i}, {'Layer': i}, {'Layer': i}
        hi = avg_hiddens[i]
        for j in range(num_layers + 1):
            hj = avg_hiddens[j]
            r_m1[str(j)] = np.mean((hi - hj)**2)
            r_m2[str(j)] = 1.0 - (np.dot(hi, hj) / (np.linalg.norm(hi) * np.linalg.norm(hj) + 1e-9))
            r_m3[str(j)] = np.linalg.norm(hj - hi) / (np.linalg.norm(hi) + 1e-9)
            r_m4[str(j)] = pearsonr(hi, hj)[0] if np.std(hi)>0 and np.std(hj)>0 else 0
            r_m5[str(j)] = np.linalg.norm(hi - hj, ord=1)
            r_m6[str(j)] = np.linalg.norm(hi - hj, ord=np.inf)
            r_m7[str(j)] = np.var(hj) / (np.var(hi) + 1e-9)
            r_m8[str(j)] = pearsonr(hi, hj)[0] if np.std(hi)>0 and np.std(hj)>0 else 0
        m1_mse.append(r_m1); m2_cos.append(r_m2); m3_res.append(r_m3); m4_cka.append(r_m4)
        m5_l1.append(r_m5); m6_linf.append(r_m6); m7_var.append(r_m7); m8_pear.append(r_m8)
    m_dir = PATHS[task_name]['metrics']
    pd.DataFrame(m1_mse).to_csv(os.path.join(m_dir, "metric_01_MSE.csv"), index=False)
    pd.DataFrame(m2_cos).to_csv(os.path.join(m_dir, "metric_02_Cosine_Distance.csv"), index=False)
    pd.DataFrame(m3_res).to_csv(os.path.join(m_dir, "metric_03_Residual_Contribution.csv"), index=False)
    pd.DataFrame(m4_cka).to_csv(os.path.join(m_dir, "metric_04_CKA.csv"), index=False)
    pd.DataFrame(m5_l1).to_csv(os.path.join(m_dir, "metric_05_L1_Distance.csv"), index=False)
    pd.DataFrame(m6_linf).to_csv(os.path.join(m_dir, "metric_06_L_Infinity.csv"), index=False)
    pd.DataFrame(m7_var).to_csv(os.path.join(m_dir, "metric_07_Variance_Ratio.csv"), index=False)
    pd.DataFrame(m8_pear).to_csv(os.path.join(m_dir, "metric_08_Pearson_Correlation.csv"), index=False)
    num_eval_samples = 20
    eval_samples = samples[:num_eval_samples]
    kl_accum = np.zeros(num_layers)
    ll_accum = np.zeros(num_layers)
    for sample in tqdm(eval_samples, desc="KL & LogitLens (Batch)"):
        messages = [{"role": "user", "content": sample["prompt"]}]
        prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            base_logits = model(**inputs).logits[0, -1, :].cpu()
            base_probs = F.softmax(base_logits.float(), dim=-1).numpy()
            base_argmax = torch.argmax(base_logits).item()
            outputs = model(**inputs, output_hidden_states=True)
            hiddens = outputs.hidden_states
        for i in range(num_layers):
            h_i = hiddens[i][0, -1, :].to(model.device) 
            h_norm = norm_layer(h_i)
            logits_i = lm_head(h_norm).cpu()
            prob_correct_token = F.softmax(logits_i.float(), dim=-1)[base_argmax].item()
            ll_accum[i] += prob_correct_token
            noise = torch.randn_like(h_i) * 0.1
            h_noisy = norm_layer(h_i + noise)
            logits_noisy = lm_head(h_noisy).cpu()
            noisy_probs = F.softmax(logits_noisy.float(), dim=-1).detach().numpy()
            kl_val = entropy(base_probs, noisy_probs + 1e-9)
            kl_accum[i] += kl_val
    kl_data = [{'Layer': i, 'Value': kl_accum[i] / num_eval_samples} for i in range(num_layers)]
    ll_data = [{'Layer': i, 'Value': ll_accum[i] / num_eval_samples} for i in range(num_layers)]
    pd.DataFrame(kl_data).to_csv(os.path.join(m_dir, "metric_12_KL_noise.csv"), index=False)
    pd.DataFrame(ll_data).to_csv(os.path.join(m_dir, "metric_13_LogitLens.csv"), index=False)
    dummy_moe = pd.DataFrame([{'Layer': i, 'Value': 0.0} for i in range(num_layers)])
    dummy_moe.to_csv(os.path.join(m_dir, "metric_09_Router_Entropy.csv"), index=False)
    dummy_moe.to_csv(os.path.join(PATHS[task_name]['data-free'], "metric_10_Router_Weights.csv"), index=False)

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.eval()
    num_layers = len(get_layers(model))
    calculate_svd_entropy(model, num_layers, PATHS)
    for task in TASKS:
        samples = load_dataset(task)
        if samples:
            run_task_pipeline(model, tokenizer, task, samples)

if __name__ == "__main__":
    main()
