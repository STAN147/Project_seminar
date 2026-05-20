import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_path = r""
dataset_path = r""
benchmark_dir = r""

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

def extract_pure_tensor(x):
    """Рекурсивно достает тензор из любых вложенных кортежей/списков"""
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)):
        return extract_pure_tensor(x[0])
    raise ValueError(f"Не удалось найти тензор в структуре: {type(x)}")


def hook_fn(module, args, output):
    pure_input_tensor = extract_pure_tensor(args)
    if isinstance(output, tuple):
        new_output = list(output)
        new_output[0] = pure_input_tensor
        return tuple(new_output)
    return pure_input_tensor


def test_model(model, data, limit, log_filename, layer_name):
    correct = 0
    log_path = os.path.join(benchmark_dir, log_filename)
    with open(log_path, "w", encoding="utf-8") as f:
        def log_and_print(text):
            """Печатает в консоль и сразу пишет в файл"""
            print(text)
            f.write(text + "\n")
            f.flush()
        log_and_print(f"=== Тестирование: {layer_name} ===")
        for i in range(limit):
            item = data[i]
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
            correct_answer = item.get('answerKey')
            if answer and correct_answer and answer[0] == correct_answer:
                correct += 1
            log_and_print(f"Вопрос {i + 1:02d}/{limit} | Ответ: {answer[0] if answer else '-'} | Правильный: {correct_answer}")
        accuracy = (correct / limit) * 100 if limit > 0 else 0
        log_and_print(f"\nРезультат: {correct}/{limit} ({accuracy:.1f}%)")
        log_and_print("=" * 50)
    return accuracy

baseline_accuracy = 70.7 

target_layers = range(24)
results = {}

for layer_idx in target_layers:
    print(f"\n>>> Приступаем к отключению слоя {layer_idx}...")
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    acc = test_model(model, data, limit, f"{layer_idx}.txt", f"ОТКЛЮЧЁН СЛОЙ {layer_idx}")
    results[layer_idx] = acc
    handle.remove()
    torch.cuda.empty_cache()
summary_text = "\n" + "=" * 60 + "\nИТОГОВАЯ СВОДКА РЕЗУЛЬТАТОВ\n" + "=" * 60 
summary_text += f"\nОригинальная модель (известный бейслайн): {baseline_accuracy:.1f}%\n\nОтключение слоёв:\n"
for layer_idx, acc in results.items():
    diff = acc - baseline_accuracy
    sign = "+" if diff > 0 else ""
    summary_text += f"  Слой {layer_idx:2d}: {acc:5.1f}% ({sign}{diff:+.1f}%)\n"
print(summary_text)
with open(os.path.join(benchmark_dir, "summary.txt"), "w", encoding="utf-8") as f:
    f.write(summary_text)

print(f"\nВсе логи успешно сохранены в папку: {benchmark_dir}")
