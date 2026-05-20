import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model_path = r""
dataset_path = r""

tokenizer = AutoTokenizer.from_pretrained(model_path)
df = pd.read_parquet(dataset_path)
data = df.to_dict('records')
limit = len(data)

print("Загружаем модель (один раз)...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"": 0},
    dtype=torch.float16,
)
model.eval()
print("Модель загружена!\n")

def disable_layer(model, layer_idx):
    """
    Заменяет указанный слой на identity-функцию (пропускает вход)
    """
    original_forward = model.model.layers[layer_idx].forward

    def identity_forward(hidden_states, *args, **kwargs):
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        return hidden_states

    model.model.layers[layer_idx].forward = identity_forward
    return original_forward

def restore_layer(model, layer_idx, original_forward):
    """
    Восстанавливает оригинальный forward слой
    """
    model.model.layers[layer_idx].forward = original_forward

def test_model(model, data, limit, layer_name="Без отключения"):
    correct = 0

    print(f"\n=== Тестирование: {layer_name} ===")

    for i in range(limit):
        item = data[i]
        if 'question' in item:
            question = item['question']
        elif 'question_text' in item:
            question = item['question_text']
        else:
            question = str(item.get('question', ''))
        if 'choices' in item:
            choices = item['choices']
            if isinstance(choices, dict) and 'label' in choices and 'text' in choices:
                labels = choices['label']
                texts = choices['text']
            elif isinstance(choices, list):
                labels = [chr(65 + i) for i in range(len(choices))]
                texts = [choice.get('text', str(choice)) for choice in choices]
            else:
                labels = [chr(65 + i) for i in range(len(choices))]
                texts = choices
        else:
            labels = [chr(65 + i) for i in range(5)]
            texts = [item.get(f'choice_{chr(65 + i)}', '') for i in range(5)]
        prompt = f"Question: {question}\nOptions:\n"
        for label, text in zip(labels, texts):
            prompt += f"{label}. {text}\n"
        prompt += "\nSelect the correct option. Answer with a single letter (A, B, C, D, or E)."
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
        if 'answerKey' in item:
            correct_answer = item['answerKey']
        elif 'answer' in item:
            correct_answer = item['answer']
        elif 'label' in item:
            correct_answer = item['label']
        else:
            correct_answer = None
        if answer and correct_answer and answer[0] == correct_answer:
            correct += 1
        print(f"Вопрос {i + 1:02d}/{limit} | Ответ: {answer[0] if answer else '-'} | Правильный: {correct_answer}")
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

print("\n" + "=" * 60)
print("ИТОГОВАЯ СВОДКА РЕЗУЛЬТАТОВ")
print("=" * 60)
print(f"Оригинальная модель: {baseline_accuracy:.1f}%")
print("\nОтключение слоёв:")
for layer_idx, acc in results.items():
    print(f"  Слой {layer_idx:2d}: {acc:5.1f}%")
