"""Console and file-dialog input helpers for STEP-ROM."""

from __future__ import annotations

from pathlib import Path

import tkinter as tk
from tkinter import filedialog

from .configuration import PipelineInput, UserParameter


def choose_project_archive() -> Path:
    """Ask the user to choose an Ansys Workbench archive via a file dialog.

    If a graphical dialog is unavailable, the function falls back to a console
    path prompt so that the same main.py remains usable in headless sessions.
    """

    selected = ""
    try:
        root = tk.Tk()
        root.withdraw()
        root.update()
        selected = filedialog.askopenfilename(
            title="Выберите проект Ansys Workbench (*.wbpz)",
            filetypes=[("Ansys Workbench archive", "*.wbpz"), ("All files", "*.*")],
        )
        root.destroy()
    except Exception:
        selected = ""

    if not selected:
        selected = input("Укажите путь к архиву проекта .wbpz: ").strip().strip('"')

    archive = Path(selected).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Файл проекта не найден: {archive}")
    return archive


def ask_yes_no(prompt: str) -> bool:
    while True:
        value = input(f"{prompt} [y/n]: ").strip().lower()
        if value in {"y", "yes", "д", "да"}:
            return True
        if value in {"n", "no", "н", "нет"}:
            return False
        print("Введите 'y'/'n' или 'да'/'нет'.")


def ask_int(prompt: str, minimum: int = 1) -> int:
    while True:
        raw = input(f"{prompt}: ").strip()
        try:
            value = int(raw)
        except ValueError:
            print("Введите целое число.")
            continue
        if value < minimum:
            print(f"Значение должно быть >= {minimum}.")
            continue
        return value


def ask_float(prompt: str) -> float:
    while True:
        raw = input(f"{prompt}: ").strip().replace(",", ".")
        try:
            return float(raw)
        except ValueError:
            print("Введите число.")


def collect_pipeline_input() -> PipelineInput:
    archive = choose_project_archive()
    is_parameterized = ask_yes_no("Используется параметризованная модель?")

    parameters: list[UserParameter] = []
    if is_parameterized:
        start_number = ask_int("С какого номера начинаются параметры P", minimum=1)
        count = ask_int("Количество параметров", minimum=1)
        for offset in range(count):
            name = f"P{start_number + offset}"
            value = ask_float(f"Значение параметра {name}")
            parameters.append(UserParameter(name=name, value=value))

    return PipelineInput(
        project_archive=archive,
        is_parameterized=is_parameterized,
        parameters=parameters,
    )
