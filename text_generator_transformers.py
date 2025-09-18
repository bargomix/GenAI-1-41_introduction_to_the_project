#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
text_gen_pipeline.py
Генерация текста с HuggingFace transformers.
"""

import argparse
import sys

import torch
from transformers import pipeline

# Определяем устройство: GPU если доступно, иначе CPU
DEVICE = 0 if torch.cuda.is_available() else -1

# Модель по умолчанию
DEFAULT_MODEL = "sberbank-ai/rugpt3medium_based_on_gpt2"

def build_generator(model_name: str = DEFAULT_MODEL):
    """
    Создаёт пайплайн генерации текста с помощью HuggingFace.
    model_name — имя модели из репозитория HF.
    DEVICE — выбирается автоматически (GPU/CPU).
    """
    return pipeline("text-generation", model = model_name, device = DEVICE)

def _wc(s: str) -> int:
    """Подсчёт количества слов в строке."""
    return len(s.split())

def _normalize(s: str) -> str:
    """Убираем лишние пробелы и переносы строк."""
    return " ".join(s.split())

def generate_until_ok(gen, prompt: str, min_words: int, max_words: int, max_attempts: int = 5):
    """
    Пытается несколько раз, пока текст не попадёт в диапазон слов.
    Если не получилось — возвращает последний вариант.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Промпт не должен быть пустым.")

    min_new = 48
    max_new = 90

    last_text, last_wc = "", 0

    for _ in range(max_attempts):
        out = gen(
            prompt,
            min_new_tokens = min_new,
            max_new_tokens = max_new,
            do_sample = True,
            temperature = 0.85,
            top_k = 40,
            top_p = 0.92,
            no_repeat_ngram_size = 4,
            repetition_penalty = 1.2,
            pad_token_id = 50256,
            truncation=True
        )
        text = _normalize(out[0]["generated_text"])
        wc = _wc(text)

        last_text, last_wc = text, wc

        if min_words <= wc <= max_words:
            return text, wc, True

        # Адаптация: если мало слов — увеличиваем окно
        if wc < min_words:
            min_new = int(min_new * 1.2)
            max_new = int(max_new * 1.3)
        # Если много слов — уменьшаем окно
        else:
            max_new = max(int(max_new * 0.85), min_new + 16)

    return last_text, last_wc, False

def parse_args():
    """
    Разбирает аргументы командной строки.
    Поддерживаются:
      --prompt    : стартовый текст
      --model     : имя модели HuggingFace
      --min_words : минимальное количество слов
      --max_words : максимальное количество слов
    """
    p = argparse.ArgumentParser(description = "Генерация текста заданной длины (по словам).")
    p.add_argument("--prompt", default = "В далёкой галактике", help = "Стартовый текст.")
    p.add_argument("--model", default = DEFAULT_MODEL, help = "Имя модели HF.")
    p.add_argument("--min_words", type = int, default = 30, help = "Минимум слов (по умолчанию 30).")
    p.add_argument("--max_words", type = int, default = 50, help = "Максимум слов (по умолчанию 50).")
    return p.parse_args()

def main():
    """
    Главная точка входа:
      1. Считывает аргументы.
      2. Проверяет корректность диапазона слов.
      3. Создаёт генератор.
      4. Запускает генерацию.
      5. Выводит результат и проверку метрики.
    """
    args = parse_args()

    # Проверка корректности диапазона
    if args.min_words <= 0 or args.max_words <= 0:
        print("[Ошибка] Количество слов должно быть положительным числом.", file = sys.stderr)
        sys.exit(1)
    if args.min_words > args.max_words:
        print("[Ошибка] min_words не может быть больше max_words.", file = sys.stderr)
        sys.exit(1)

    print(f"Device: {'CUDA:0' if DEVICE == 0 else 'CPU'}")
    gen = build_generator(args.model)

    text, wc, ok = generate_until_ok(gen, args.prompt, args.min_words, args.max_words)
    print(text)
    print(f"\nСлов: {wc} | Метрика ок: {ok}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Ошибка] {e}", file = sys.stderr)
        sys.exit(1)
