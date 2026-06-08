import os
import sys
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

# --- БЛОК ДЛЯ СОХРАНЕНИЯ TXT-ЛОГА ---
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

# Запускаем логгер до основных выводов
log_filename = os.path.join(SCRIPT_DIR, "qwen_siqa_testing_log.txt")
sys.stdout = DualLogger(log_filename)
print(f"📄 Запись лога тестирования начата в файл: {log_filename}\n")

# Пути к файлам
model_path = os.path.abspath(os.path.join(BASE_DIR, "models", "Qwen"))
dataset_path = os.path.abspath(os.path.join(BASE_DIR, "datasets", "siqa_500.csv"))

# Загрузка токенизатора и корректное чтение CSV
tokenizer = AutoTokenizer.from_pretrained(model_path)
df = pd.read_csv(dataset_path)
data = df.to_dict('records')
limit = len(data)

print("Загружаем модель Qwen (в 4-битном режиме с жесткой посадкой на GPU)...")

# 1. Создаем конфиг квантизации (ужимаем веса в 4 бита)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# 2. Загружаем модель, принудительно заталкивая ВСЕ слои на GPU 0
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"": 0}, # Жестко фиксируем на GPU, запрещая offload на CPU/диск
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
)
model.eval()
print("Модель успешно загружена!\n")

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

    # Карта перевода числовых меток SIQA в буквенные опции
    label_map = {'1': 'A', '2': 'B', '3': 'C', 1: 'A', 2: 'B', 3: 'C'}

    for i in range(limit):
        item = data[i]
        
        context = item.get('context', '')
        question = item.get('question', '')
        
        # SIQA содержит строго 3 варианта ответа
        texts = [item.get('answerA', ''), item.get('answerB', ''), item.get('answerC', '')]
        labels = ['A', 'B', 'C']
        
        # Формируем структурированный промпт
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
        
        # Конвертируем метку датасета (1, 2, 3) в букву (A, B, C)
        raw_label = item.get('label', '')
        correct_answer = label_map.get(raw_label, None)
        
        if answer and correct_answer and answer[0] == correct_answer:
            correct += 1
            
        print(f"Вопрос {i + 1:03d}/{limit} | Ответ: {answer[0] if answer else '-'} | Правильный: {correct_answer}")
        
    accuracy = (correct / limit) * 100 if limit > 0 else 0
    print(f"\nРезультат: {correct}/{limit} ({accuracy:.1f}%)")
    print("=" * 50)
    return accuracy

# --- Вычисление базовой точности ---
baseline_accuracy = test_model(model, data, limit, "ОРИГИНАЛЬНАЯ МОДЕЛЬ")

results = {}
total_layers = len(model.model.layers)
print(f"\nВсего слоёв в модели: {total_layers}")

# --- Цикл последовательной абляции слоев ---
for layer_idx in range(total_layers):
    original_forward = disable_layer(model, layer_idx)
    accuracy = test_model(model, data, limit, f"ОТКЛЮЧЁН СЛОЙ {layer_idx}")
    results[layer_idx] = accuracy
    restore_layer(model, layer_idx, original_forward)
    torch.cuda.empty_cache()

# --- Вывод сводки в консоль и логгер ---
print("\n" + "=" * 60)
print("ИТОГОВАЯ СВОДКА РЕЗУЛЬТАТОВ")
print("=" * 60)
print(f"Оригинальная модель: {baseline_accuracy:.1f}%")
print("\nОтключение слоёв:")
for layer_idx, acc in results.items():
    print(f"  Слой {layer_idx:2d}: {acc:5.1f}%")

# --- СОХРАНЕНИЕ МЕТРИК В CSV ДЛЯ ИНТЕГРАЦИИ В LASSO ---
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

print("\n" + "=" * 60)
print(f"✅ Метрики успешно сохранены в CSV-файл:\n{output_path}")
print("=" * 60)
