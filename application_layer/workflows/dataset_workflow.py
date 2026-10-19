# ============================================================
# МОДУЛЬ: application_layer/workflows/dataset_workflow.py
#
# Назначение:
# - интерактивный выбор источника данных;
# - загрузка существующего файла;
# - генерация синтетического датасета;
# - генерация данных через SDV.
# ============================================================


import pandas as pd

from data_layer.data_loader import DataLoader


def choose_dataset() -> tuple[pd.DataFrame, str | None]:
    """
    Спрашивает пользователя, какой датасет использовать.
    """
    storage_dir = DataLoader.STORAGE_DIR
    storage_dir.mkdir(parents=True, exist_ok=True)

    files = [
        file for file in storage_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() in (".csv", ".xlsx", ".xls", ".json")
    ]

    print("\nВыберите источник данных:")
    print("  1 — Использовать существующий файл из storage/")
    print("  2 — Сгенерировать новый синтетический датасет")
    print("  3 — Сгенерировать новый датасет через SDV")

    choice = input("Ваш выбор [1/2/3]: ").strip()

    if choice == "1":
        return _load_existing_dataset(files)

    if choice == "2":
        return _generate_synthetic_dataset()

    if choice == "3":
        return _generate_sdv_dataset(files)

    print("Некорректный выбор.")
    return choose_dataset()


def _load_existing_dataset(files: list) -> tuple[pd.DataFrame, str]:
    """
    Загружает выбранный пользователем файл.
    """
    if not files:
        print("В папке storage/ нет доступных файлов.")
        return choose_dataset()

    print("\nДоступные файлы:")

    for index, file in enumerate(files, start=1):
        print(f"  {index}. {file.name}")

    file_number = input("Введите номер файла: ").strip()

    try:
        selected_file = files[int(file_number) - 1]
    except (ValueError, IndexError):
        print("Некорректный номер файла.")
        return choose_dataset()

    df = DataLoader.load(str(selected_file))

    print(f"Загружен файл: {selected_file}")

    return df, str(selected_file)


def _generate_synthetic_dataset() -> tuple[pd.DataFrame, str]:
    """
    Генерирует синтетический датасет.
    """
    n_samples = input("Количество строк [5000]: ").strip()
    n_samples = int(n_samples) if n_samples else 5000

    filename = input("Имя файла [synthetic_data]: ").strip()
    filename = filename or "synthetic_data"

    df = DataLoader.generate_synthetic(
        n_samples=n_samples,
        seed=42,
        save=True,
        filename=filename,
        fmt="csv"
    )

    source = str(DataLoader.STORAGE_DIR / f"{filename}.csv")

    return df, source


def _generate_sdv_dataset(files: list) -> tuple[pd.DataFrame, str]:
    """
    Генерирует синтетический датасет через SDV.
    """
    if not files:
        print("Для SDV нужен хотя бы один файл в storage/.")
        return choose_dataset()

    print("\nВыберите исходный файл:")

    for index, file in enumerate(files, start=1):
        print(f"  {index}. {file.name}")

    file_number = input("Введите номер файла: ").strip()

    try:
        selected_file = files[int(file_number) - 1]
    except (ValueError, IndexError):
        print("Некорректный номер файла.")
        return choose_dataset()

    real_data = DataLoader.load(str(selected_file))

    n_samples = input("Количество строк [5000]: ").strip()
    n_samples = int(n_samples) if n_samples else 5000

    filename = input("Имя нового файла [sdv_synthetic_data]: ").strip()
    filename = filename or "sdv_synthetic_data"

    df = DataLoader.generate_with_sdv(
        real_data=real_data,
        n_samples=n_samples,
        seed=42,
        save=True,
        filename=filename,
        fmt="csv"
    )

    source = str(DataLoader.STORAGE_DIR / f"{filename}.csv")

    return df, source


print("Модуль application_layer.workflows.dataset_workflow определён.")