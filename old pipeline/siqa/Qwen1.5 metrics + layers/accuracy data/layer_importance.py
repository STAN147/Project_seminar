import os
import sys
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()

log_filename = os.path.join(SCRIPT_DIR, "qwen_siqa_testing_log.txt")
sys.stdout = DualLogger(log_filename)

model_path = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "Qwen"))
dataset_path = os.path.abspath(os.path.join(BASE_DIR, "..", "datasets", "siqa_500.csv"))

tokenizer = AutoTokenizer.from_pretrained(model_path)
df = pd.read_csv(dataset_path)
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

def disable_layer(model, layer_idx):
    """ Заменяет указанный слой на identity-функцию (пропускает вход) """
    original_forward = model.model.layers[layer_idx].forward

    def identity_forward(hidden_states, *args, **kwargs):
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        return hidden_states

    model.model.layers[layer_idx].forward = identity_forward
    return original_forward

def restore_layer(model, layer_idx, original_forward):
    """ Восстанавливает оригинальный forward слоя """
    model.model.layers[layer_idx].forward = original_forward

def test_model(model, data, limit, layer_name="Без отключения"):
    correct = 0
    print(f"\n=== Тестирование: {layer_name} ===")

    label_map = {'1': 'A', '2': 'B', '3': 'C', 1: 'A', 2: 'B', 3: 'C'}

    for i in range(limit):
        item = data[i]
        
        context = item.get('context', '')
        question = item.get('question', '')

        texts = [item.get('answerA', ''), item.get('answerB', ''), item.get('answerC', '')]
        labels = ['A', 'B', 'C']
        
        prompt = f"Context: {context}\nQuestion: {question}\nOptions:\n"
        for label, text in zip(labels, texts):
            prompt += f"{label}. {text}\n"
        prompt += "\nSelect the correct option. Answer with a single letter (A, B, or C)."
        
        messages = [
            {"role": "system", "content": "You are a logical AI. Output only the letter of the correct answer."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=2,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        answer = tokenizer.decode(output_ids[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        
        raw_label = item.get('label', '')
        correct_answer = label_map.get(raw_label, None)
        
        if answer and correct_answer and answer[0] == correct_answer:
            correct += 1
            
        print(f"Вопрос {i + 1:03d}/{limit} | Ответ: {answer[0] if answer else '-'} | Правильный: {correct_answer}")
        
    accuracy = (correct / limit) * 100 if limit > 0 else 0
    print(f"\nРезультат: {correct}/{limit} ({accuracy:.1f}%)")
    print("=" * 50)
    return accuracy

baseline_accuracy = test_model(model, data, limit, "ОРИГИНАЛЬНАЯ МОДЕЛЬ")

results = {}
total_layers = len(model.model.layers)
print(f"\nВсего слоёв в модели: {total_layers}")

for layer_idx in range(total_layers):
    original_forward = disable_layer(model, layer_idx)
    accuracy = test_model(model, data, limit, f"ОТКЛЮЧЁН СЛОЙ {layer_idx}")
    results[layer_idx] = accuracy
    restore_layer(model, layer_idx, original_forward)
    torch.cuda.empty_cache()

print(f"Оригинальная модель: {baseline_accuracy:.1f}%")
print("\nОтключение слоёв:")
for layer_idx, acc in results.items():
    print(f"  Слой {layer_idx:2d}: {acc:5.1f}%")
output_filename = "phi_tiny_siqa_ablations.csv"
output_path = os.path.join(SCRIPT_DIR, output_filename)
csv_data = []
for layer_idx, acc in results.items():
    drop = acc - baseline_accuracy
    csv_data.append({
        "Layer": layer_idx,
        "Accuracy": acc,
        "Ablation_Drop": round(drop, 2)
    })
df_results = pd.DataFrame(csv_data)
df_results.to_csv(output_path, index=False)
