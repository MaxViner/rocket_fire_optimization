# ============================================================
# МОДУЛЬ: application_layer/workflows/batch_workflow.py
#
# Назначение:
# - пакетная обработка готовых сценариев;
# - сохранение входных и выходных данных в storage.
# ============================================================


import pandas as pd

from application_layer.api import ApplicationAPI
from data_layer.data_loader import DataLoader


def run_batch_processing(api: ApplicationAPI):
    """
    Выполняет пакетную обработку тестовых сценариев.
    """
    print("\nПакетная обработка готовых сценариев...")

    batch_input = pd.DataFrame([
        {
            "H": 800, "D": 2000, "V": 240, "N": 4,
            "C": "stronghold", "T": "destroy",
            "W_dir": 90, "W_spd": 8, "Temp": 15
        },
        {
            "H": 1200, "D": 3500, "V": 260, "N": 7,
            "C": "apc", "T": "damage",
            "W_dir": 180, "W_spd": 12, "Temp": 10
        },
        {
            "H": 600, "D": 4500, "V": 220, "N": 10,
            "C": "infantry", "T": "suppress",
            "W_dir": 270, "W_spd": 5, "Temp": 25
        }
    ])

    input_path = DataLoader.STORAGE_DIR / "batch_input.csv"
    output_path = DataLoader.STORAGE_DIR / "batch_output.csv"

    batch_input.to_csv(input_path, index=False)

    batch_result = api.batch_predict(
        input_path=str(input_path),
        output_path=str(output_path)
    )

    print(f"Обработано строк: {len(batch_result)}")
    print(f"Результат сохранён: {output_path}")


print("Модуль application_layer.workflows.batch_workflow определён.")