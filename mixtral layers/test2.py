import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

# ---------- НАСТРОЙКИ ----------
MODEL_PATH = "../models/mixtral"

# Явно указываем, какие GPU использовать (0 и 1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print("Используемые GPU:", os.environ["CUDA_VISIBLE_DEVICES"])

# ---------- ТОКЕНИЗАТОР ----------
print("Загрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, local_files_only=True)
print("Токенизатор загружен.")

# ---------- 4-БИТНОЕ КВАНТОВАНИЕ С РАСШИРЕННЫМИ НАСТРОЙКАМИ ----------
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # Включаем 4-битный режим
    bnb_4bit_compute_dtype=torch.float16,   # Вычисления в 16-битном формате
    bnb_4bit_use_double_quant=True,         # Включаем двойное квантование для экономии памяти
    bnb_4bit_quant_type="nf4",              # Используем более качественный тип квантования
    llm_int8_enable_fp32_cpu_offload=True   # Разрешаем выгрузку на CPU при нехватке памяти
)

# ---------- ЗАГРУЗКА МОДЕЛИ С РУЧНЫМИ ЛИМИТАМИ ПАМЯТИ ----------
print("Загрузка модели (может занять несколько минут)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",                      # Автоматическое распределение по устройствам
    max_memory={0: "20GiB", 1: "20GiB"},    # <--- КЛЮЧЕВОЙ МОМЕНТ: лимит по 20 ГБ на каждую GPU
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    local_files_only=True
)
print("Модель загружена!")
print(f"Карта устройств: {model.hf_device_map}")