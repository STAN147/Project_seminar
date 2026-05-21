import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

model_path = os.path.abspath(os.path.join(BASE_DIR, "models", "Qwen"))
dataset_path = os.path.abspath(os.path.join(BASE_DIR, "datasets", "dev_rand_split.jsonl"))

tokenizer = AutoTokenizer.from_pretrained(model_path)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"": 0},
    quantization_config=quantization_config,
)

data = []
with open(dataset_path, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

limit = len(data)
correct = 0

for i in range(limit):
    item = data[i]
    question = item['question']
    choices = item['choices']
    prompt = f"Question: {question}\nOptions:\n"
    for label, text in zip(choices['label'], choices['text']):
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
    correct_answer = item['answerKey']
    if answer and answer.startswith(correct_answer):
        correct += 1
    print(f"Вопрос {i+1:02d}/{limit} | Ответ модели: {answer[0] if answer else '-'} | Правильный: {correct_answer}")

accuracy = (correct / limit) * 100
print(f"\n\n--- ИТОГОВЫЕ РЕЗУЛЬТАТЫ ---")
print(f"Правильных ответов: {correct} из {limit}")
print(f"Точность (Accuracy): {accuracy:.1f}%")
