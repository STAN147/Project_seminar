#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import os

# ---------- НАСТРОЙКИ ----------
MODEL_PATH = "../models/mixtral"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Обе карты

print("Используемые GPU:", os.environ["CUDA_VISIBLE_DEVICES"])

# ---------- ТОКЕНИЗАТОР С ОТЛАДКОЙ ----------
print("Загрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, local_files_only=True)
print("Токенизатор загружен.")

# ---------- 4-БИТНОЕ КВАНТОВАНИЕ ----------
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# ---------- ЗАГРУЗКА МОДЕЛИ С ПРОСТЫМ device_map ----------
print("Загрузка модели (будет виден прогресс)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",                     # Позволяет transformers самому распределить
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    local_files_only=True                  # Не ходим в интернет, только локальные файлы
)
print("Модель загружена!")
print(f"Карта устройств: {model.hf_device_map}")

# ---------- ТЕСТОВЫЙ ВОПРОС (один для начала) ----------
def ask_question(question):
    prompt = f"[INST] {question} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
    input_length = inputs.input_ids.shape[1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

print("\nТестовый вопрос: What is the capital of France?")
print("Ответ:", ask_question("What is the capital of France?"))