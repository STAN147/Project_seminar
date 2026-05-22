import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from accelerate import infer_auto_device_map

# ---------- НАСТРОЙКИ ----------
MODEL_PATH = "../models/mixtral"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Указываем обе карты
print("Начинаем загрузку...")

# 1. Загружаем токенизатор (оставляем медленный для совместимости)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

# 2. Настраиваем 4-битное квантование с флагами для экономии памяти
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 3. Получаем конфигурацию модели и создаём "пустую" модель для расчётов
config = AutoConfig.from_pretrained(MODEL_PATH)
dummy_model = AutoModelForCausalLM.from_config(config)

# 4. Явно создаём карту устройств: по 10 ГБ на каждую карту, остальное на CPU
max_memory = {0: "100GiB", 1: "100GiB", "cpu": "128GiB"}
device_map = infer_auto_device_map(
    dummy_model,
    max_memory=max_memory
)

# 5. Чистим память перед загрузкой
del dummy_model
gc.collect()
torch.cuda.empty_cache()

# 6. Загружаем модель с нашей готовой картой устройств
print("Загрузка модели (это может занять несколько минут)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map=device_map,
    low_cpu_mem_usage=True
)
print("Модель загружена!")


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