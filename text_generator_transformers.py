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
    model_name — модели из репозитория HF.
    DEVICE — выбирается автоматически (GPU/CPU).
    """
    try:
        return pipeline("text-generation", model = model_name, device=DEVICE)
    except Exception as e:
        print(
            f"[Ошибка] Модель не загрузилась: {model_name}\nПричина: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def _wc(s: str) -> int:
    """Подсчёт количества слов в строке."""
    return len(s.split())


def _normalize(s: str) -> str:
    """Убираем лишние пробелы и переносы строк."""
    return " ".join(s.split())


def generate_until_ok(
    gen,
    prompt: str,
    min_words: int,
    max_words: int,
    max_attempts: int,
    min_new_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    no_repeat_ngram_size: int,
    repetition_penalty: float,
    truncation: bool
):
    """
    Пытается несколько раз, пока текст не попадёт в диапазон слов.
    Если не получилось — возвращает последний вариант.
    Добавлен try/except
    """
    if not prompt or not prompt.strip():
        raise ValueError("Промпт не должен быть пустым.")

    min_new = min_new_tokens
    max_new = max_new_tokens

    last_text, last_wc = "", 0

    for attempt in range(1, max_attempts + 1):
        try:
            out = gen(
                prompt,
                min_new_tokens = min_new,
                max_new_tokens = max_new,
                do_sample = do_sample,
                temperature = temperature,
                top_k = top_k,
                top_p = top_p,
                no_repeat_ngram_size = no_repeat_ngram_size,
                repetition_penalty = repetition_penalty,
                pad_token_id = 50256,
                truncation = truncation,
            )
        except Exception as e:
            print(f"[Ошибка] Генерация прервалась на попытке {attempt}: {e}", file=sys.stderr)
            continue  # пробуем ещё раз

        text = _normalize(out[0]["generated_text"])
        wc = _wc(text)

        last_text, last_wc = text, wc

        if min_words <= wc <= max_words:
            return text, wc, True

        if wc < min_words:
            min_new = int(min_new * 1.2)
            max_new = int(max_new * 1.3)
        else:
            max_new = max(int(max_new * 0.85), min_new + 16)

    return last_text, last_wc, False


def parse_bool(v: str) -> bool:
    s = str(v).strip().lower()
    if s in ("true", "t", "1", "yes", "y", "on"):
        return True
    if s in ("false", "f", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("ожидалось значение типа bool: true/false")

def parse_args():
    """
    Разбирает аргументы командной строки.
    Поддерживается:
      --prompt  : стартовый текст
      --model     : имя модели HuggingFace
      --min_words : минимальное количество слов
      --max_words : максимальное количество слов
      --attempts  : колличество попыток генерации
      --min_new_tokens : минимальное колличество сгеренированных токенов
      --max_new_tokens : максимальное колличество сгеренированных токенов
      --do_sample : режим выбора токенов жадная или случайная с учетом вероятностей
      --temperature :   темпаратура сэмплинга "криативность"
      --top-k : top-k выбор (связность текста)
      --top-p : top-p выбор (связность текста более разнообразный чем с top-p)
      --no_repeat_ngram_size : запрет на повторение одинаковых токенов (зацикливание)
      --repetition_penalty : штраф за повторение (если токен уже был использован его вероятность снижается)
      --truncation : обрезка входящего текста если он больше чем может обработать модель
    """
    p = argparse.ArgumentParser(description = "Генерация текста заданной длины (по словам).")
    p.add_argument("--prompt", default ="В далёкой галактике", help = "Стартовый текст.")
    p.add_argument("--model", default = DEFAULT_MODEL, help = "Имя модели HF.")
    p.add_argument("--min_words", type = int, default = 30, help = "Минимум слов (по умолчанию 30).")
    p.add_argument("--max_words", type = int, default = 50, help = "Максимум слов (по умолчанию 50).")
    p.add_argument("--attempts", type = int, default = 5, help = "Количество попыток (по умолчанию 5).")
    p.add_argument("--min_new_tokens", type = int, default = 48, help = "Стартовое значение min_new_tokens (по умолчанию 48).")
    p.add_argument("--max_new_tokens", type = int, default = 90, help = "Стартовое значение max_new_tokens (по умолчанию 90).")
    p.add_argument("--do_sample", type = parse_bool, default = True, help = "Жадная или случайная стратегия выборки токенов {true|false}")
    p.add_argument("--temperature", type = float, default = 0.85, help = "Температура сэмплинга (по умолчанию 0.85).")
    p.add_argument("--top_k", type = int, default = 40, help = "Top-k выбор(по умолчанию 40).")
    p.add_argument("--top_p", type = float, default = 0.92, help = "Top-p (nucleus sampling)(по умолчанию 0.92).")
    p.add_argument("--no_repeat_ngram_size", type = int, default = 4, help = "Запрет повторов n-грам (по умолчанию 4).")
    p.add_argument("--repetition_penalty", type = float, default = 1.2, help = "Штраф за повторения(по умолчанию 1.2).")
    p.add_argument("--truncation", type = parse_bool, default = True, help = "Обрезание входящего текста если он больше чем может обработать модель {true|false}")
    return p.parse_args()

def main():
    args = parse_args()

    if args.min_words <= 0 or args.max_words <= 0:
        print("[Ошибка] Количество слов должно быть положительным числом.", file=sys.stderr)
        sys.exit(1)
    if args.min_words > args.max_words:
        print("[Ошибка] min_words не может быть больше max_words.", file=sys.stderr)
        sys.exit(1)

    print(f"Device: {'CUDA:0' if DEVICE == 0 else 'CPU'}")
    gen = build_generator(args.model)

    text, wc, ok = generate_until_ok(
        gen,
        args.prompt,
        args.min_words,
        args.max_words,
        args.attempts,
        args.min_new_tokens,
        args.max_new_tokens,
        args.do_sample,
        args.temperature,
        args.top_k,
        args.top_p,
        args.no_repeat_ngram_size,
        args.repetition_penalty,
        args.truncation
    )
    print(text)
    print(f"\nСлов: {wc} | Метрика ок: {ok}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Ошибка] {e}", file=sys.stderr)
        sys.exit(1)