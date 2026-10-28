"""Preguntas al operador por la terminal, antes de abrir la consola de operación."""

from __future__ import annotations

from collections.abc import Mapping

YES_ANSWERS = {"s", "si", "sí", "y", "yes"}


def ask_yes_no(question: str) -> bool:
    answer = input(f"\n{question} [s/N]: ")
    return answer.strip().lower() in YES_ANSWERS


def ask_choice(question: str, options: Mapping[str, str]) -> str:
    """Repite la pregunta hasta que la respuesta sea una de las claves de ``options``."""
    menu = "\n".join(f"  {key}  {text}" for key, text in options.items())
    while True:
        answer = input(f"\n{question}\n{menu}\nOpción: ").strip()
        if answer in options:
            return answer
        print(f"'{answer}' no es una opción.")
