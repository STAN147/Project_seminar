import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

model_path = os.path.abspath(os.path.join(BASE_DIR, "models", "Qwen"))
dataset_path = os.path.abspath(os.path.join(BASE_DIR, "datasets", "dev_rand_split.jsonl"))
benchmark_dir = os.path.abspath(os.path.join(BASE_DIR, "Qwen1.5 metrics + layers + proxy", "metric data", "metrics"))

os.makedirs(benchmark_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_path)
df = pd.read_json(dataset_path, lines=True)
data = df.to_dict('records')
limit = len(data)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"": 0},
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
)
model.eval()

def calculate_cka(x, y):
    """Считает CKA (Centered Kernel Alignment)"""
    x = x[0].float()
    y = y[0].float()
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    dot_prod = torch.norm(torch.matmul(x.T, y), p='fro') ** 2
    norm_x = torch.norm(torch.matmul(x.T, x), p='fro')
    norm_y = torch.norm(torch.matmul(y.T, y), p='fro')
    if norm_x == 0 or norm_y == 0: return 0.0
    return (dot_prod / (norm_x * norm_y)).item()

num_layers = 24
sum_mse = np.zeros((num_layers, num_layers))
sum_cos = np.zeros((num_layers, num_layers))
sum_rc = np.zeros((num_layers, num_layers))
sum_cka = np.zeros((num_layers, num_layers))
sum_l1 = np.zeros((num_layers, num_layers))
sum_l_inf = np.zeros((num_layers, num_layers))
sum_var_ratio = np.zeros((num_layers, num_layers))
sum_pearson = np.zeros((num_layers, num_layers))

sum_entropy = None 

print(f"\n3. Начинаем расчет метрик ({limit} вопросов)...")

for n in tqdm(range(limit)):
    item = data[n]
    question = item.get('question', '')
    
    if 'choices' in item:
        choices = item['choices']
        labels = choices.get('label', [chr(65 + j) for j in range(len(choices.get('text', [])))])
        texts = choices.get('text', [])
    else:
        labels = [chr(65 + j) for j in range(5)]
        texts = [item.get(f'choice_{chr(65 + j)}', '') for j in range(5)]

    prompt = f"Question: {question}\nOptions:\n"
    for label, text in zip(labels, texts):
        prompt += f"{label}. {text}\n"

    messages = [{"role": "system", "content": "You are a logical AI."}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_router_logits=True)
    hidden_states = outputs.hidden_states
    if outputs.router_logits is not None:
        num_routers = len(outputs.router_logits)
        if sum_entropy is None:
            sum_entropy = np.zeros(num_routers)
        for r_idx, router_logit in enumerate(outputs.router_logits):
            probs = F.softmax(router_logit[0].float(), dim=-1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item()
            sum_entropy[r_idx] += entropy

    for i in range(num_layers):
        h_i = hidden_states[i + 1].float()
        var_i = torch.var(h_i)
        h_i_centered = h_i - h_i.mean(dim=-1, keepdim=True)
        for j in range(num_layers):
            h_j = hidden_states[j + 1].float()
            if i == j:
                sum_mse[i, j] += 0.0
                sum_cos[i, j] += 0.0
                sum_rc[i, j] += 0.0
                sum_cka[i, j] += 1.0
                sum_l1[i, j] += 0.0
                sum_l_inf[i, j] += 0.0
                sum_var_ratio[i, j] += 1.0
                sum_pearson[i, j] += 1.0
                continue
            h_j_centered = h_j - h_j.mean(dim=-1, keepdim=True)
            sum_mse[i, j] += F.mse_loss(h_i, h_j).item()
            sum_cos[i, j] += 1.0 - F.cosine_similarity(h_i, h_j, dim=-1).mean().item()
            sum_rc[i, j] += (torch.norm(h_j - h_i, dim=-1) / (torch.norm(h_i, dim=-1) + 1e-9)).mean().item()
            sum_cka[i, j] += calculate_cka(h_i, h_j)
            sum_l1[i, j] += F.l1_loss(h_i, h_j).item()
            sum_l_inf[i, j] += torch.max(torch.abs(h_i - h_j)).item()
            sum_var_ratio[i, j] += (torch.var(h_j) / (var_i + 1e-9)).item()
            sum_pearson[i, j] += F.cosine_similarity(h_i_centered, h_j_centered, dim=-1).mean().item()
    del outputs, hidden_states
    torch.cuda.empty_cache()

def save_matrix_to_csv(matrix, filename):
    filepath = os.path.join(benchmark_dir, filename)
    df_matrix = pd.DataFrame(matrix / limit)
    layer_names = [f"Layer_{i}" for i in range(num_layers)]
    df_matrix.columns = layer_names
    df_matrix.index = layer_names
    df_matrix.to_csv(filepath)

def save_vector_to_csv(vector, filename):
    filepath = os.path.join(benchmark_dir, filename)
    df_vector = pd.DataFrame(vector / limit, columns=['Avg_Router_Entropy'])
    df_vector.index = [f"MoE_Layer_{i}" for i in range(len(vector))]
    df_vector.to_csv(filepath, index_label='Layer')

save_matrix_to_csv(sum_mse, "metric_01_MSE.csv")
save_matrix_to_csv(sum_cos, "metric_02_Cosine_Distance.csv")
save_matrix_to_csv(sum_rc, "metric_03_Residual_Contribution.csv")
save_matrix_to_csv(sum_cka, "metric_04_CKA.csv")
save_matrix_to_csv(sum_l1, "metric_05_L1_Distance.csv")
save_matrix_to_csv(sum_l_inf, "metric_06_L_Infinity.csv")
save_matrix_to_csv(sum_var_ratio, "metric_07_Variance_Ratio.csv")
save_matrix_to_csv(sum_pearson, "metric_08_Pearson_Correlation.csv")
if sum_entropy is not None:
    save_vector_to_csv(sum_entropy, "metric_09_Router_Entropy.csv")
print(f"\nАнализ завершен! 9 CSV файлов успешно сохранены в папке {benchmark_dir}.")
