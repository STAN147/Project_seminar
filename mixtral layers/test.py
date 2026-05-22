#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы модели Mixtral-8x7B-Instruct-v0.1
Загружает модель из локальной папки ./mixtral и задаёт 10 вопросов.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time

# ---------- НАСТРОЙКИ ----------
MODEL_PATH = "../models/phi-tiny"  # путь к скачанной модели
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используется устройство: {DEVICE}")

# ---------- ЗАГРУЗКА МОДЕЛИ ----------
print("Загрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,               # Включаем 4-битный режим
    bnb_4bit_compute_dtype=torch.float16, # Вычисления в 16-битном формате
)
print("Загрузка модели (это может занять несколько минут)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)


model.eval()
print("Модель загружена!\n")


# ---------- ФУНКЦИЯ ДЛЯ ГЕНЕРАЦИИ ОТВЕТА ----------
def ask_question(question, max_new_tokens=256):
    """Отправляет вопрос модели Mixtral Instruct и возвращает ответ."""
    # Mixtral Instruct использует формат: [INST] вопрос [/INST]
    prompt = f"[INST] {question} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Декодируем только сгенерированную часть (после промпта)
    input_length = inputs.input_ids.shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return response.strip()


# ---------- ТЕСТОВЫЕ ВОПРОСЫ (10 штук, разной сложности) ----------
questions = [
    "What is the capital of France?",
    "Explain the concept of 'machine learning' in one sentence.",
    "If a train travels 120 km in 2 hours, what is its average speed?",
    "What does the following Python code print? `print(2 ** 3)`",
    "Who wrote 'Romeo and Juliet'?",
    "What is the primary function of a GPU?",
    "Solve for x: 2x + 5 = 15",
    "What is the difference between supervised and unsupervised learning?",
    "Why is the sky blue?",
    "If a store has a 20% off sale, and an item costs $50, what is the sale price?"
]

print("=" * 60)
print("ЗАПУСК ТЕСТА: 10 вопросов из бенчмарка")
print("=" * 60)

for i, q in enumerate(questions, 1):
    print(f"\nВопрос {i}: {q}")
    print("Ответ: ", end="", flush=True)
    start = time.time()
    answer = ask_question(q)
    elapsed = time.time() - start
    print(f"{answer}\n(время: {elapsed:.1f} сек.)")
    print("-" * 60)

print("\n✅ Тест завершён.")