# ============================================================
# МОДУЛЬ: application_layer/workflows/user_input_workflow.py
#
# Назначение:
# - выбор режима: тестовый пример или ручной ввод;
# - ввод и валидация диапазонов условий;
# - ввод погодных условий;
# - ввод требований к миссии.
# ============================================================


from config import CONFIG


TEST_APPLICATION_CONDITIONS = {
    "H": {
        "min": 100,
        "max": 1000,
        "step": 25
    },
    "D": {
        "min": 2000,
        "max": 6000,
        "step": 50
    },
    "V": {
        "min": 160,
        "max": 280,
        "step": 5
    }
}


TEST_WEATHER = {
    "dir": 90,
    "spd": 10,
    "temp": 22
}


TEST_MISSION_REQUIREMENTS = {
    "target_type": "stronghold",
    "effect_type": "destroy",
    "p_required": 0.90
}


def choose_conditions_mode() -> tuple[dict, dict, dict]:
    """
    Выбирает режим задания условий:
    тестовый пример или ручной ввод.
    """
    print("\nВыбор режима задания условий боевого применения:")
    print("  1 — Использовать готовый тестовый пример")
    print("  2 — Ввести условия вручную")

    mode = input("Ваш выбор [1/2]: ").strip()

    if mode == "2":
        return get_manual_conditions()

    print("\nИспользуется готовый тестовый пример.")

    print("\nДиапазон условий боевого применения:")
    print(f"  H: {TEST_APPLICATION_CONDITIONS['H']}")
    print(f"  D: {TEST_APPLICATION_CONDITIONS['D']}")
    print(f"  V: {TEST_APPLICATION_CONDITIONS['V']}")

    print("\nУсловия окружающей среды:")
    print(f"  {TEST_WEATHER}")

    print("\nТребования к миссии:")
    print(f"  {TEST_MISSION_REQUIREMENTS}")

    return (
        TEST_APPLICATION_CONDITIONS,
        TEST_WEATHER,
        TEST_MISSION_REQUIREMENTS
    )


def get_manual_conditions() -> tuple[dict, dict, dict]:
    """
    Запрашивает диапазоны условий, погоду и требования к миссии.
    """
    constraints = CONFIG["constraints"]

    print("\nЗадание диапазона условий боевого применения...")

    application_conditions = {
        "H": input_range(
            parameter_name="Высота пуска H, м",
            default_min=100,
            default_max=2000,
            default_step=250,
            allowed_min=constraints["H"]["min"],
            allowed_max=constraints["H"]["max"]
        ),
        "D": input_range(
            parameter_name="Дальность до цели D, м",
            default_min=1000,
            default_max=6000,
            default_step=500,
            allowed_min=constraints["D"]["min"],
            allowed_max=constraints["D"]["max"]
        ),
        "V": input_range(
            parameter_name="Скорость V, м/с",
            default_min=160,
            default_max=280,
            default_step=20,
            allowed_min=constraints["V"]["min"],
            allowed_max=constraints["V"]["max"]
        )
    }

    print("\nУсловия окружающей среды:")

    weather = {
        "dir": float(
            input("Направление ветра W_dir, град [90]: ").strip() or 90
        ),
        "spd": float(
            input("Скорость ветра W_spd, м/с [10]: ").strip() or 10
        ),
        "temp": float(
            input("Температура Temp, °C [20]: ").strip() or 20
        )
    }

    print("\nЗадание требований к миссии...")

    mission_requirements = input_mission_requirements()

    return application_conditions, weather, mission_requirements


def input_range(
    parameter_name: str,
    default_min: float,
    default_max: float,
    default_step: float,
    allowed_min: float,
    allowed_max: float
) -> dict:
    """
    Запрашивает диапазон и шаг с проверкой корректности.
    """
    print(f"\nПараметр: {parameter_name}")
    print(f"Допустимый диапазон: {allowed_min} — {allowed_max}")

    while True:
        min_value = input(
            f"  Минимальное значение [{default_min}]: "
        ).strip()

        max_value = input(
            f"  Максимальное значение [{default_max}]: "
        ).strip()

        step_value = input(
            f"  Шаг [{default_step}]: "
        ).strip()

        try:
            min_value = float(min_value) if min_value else default_min
            max_value = float(max_value) if max_value else default_max
            step_value = float(step_value) if step_value else default_step
        except ValueError:
            print("  Ошибка: введите числовые значения.")
            continue

        if min_value < allowed_min or max_value > allowed_max:
            print("  Ошибка: значения выходят за допустимый диапазон.")
            continue

        if min_value > max_value:
            print("  Ошибка: минимальное значение больше максимального.")
            continue

        if step_value <= 0:
            print("  Ошибка: шаг должен быть положительным.")
            continue

        return {
            "min": min_value,
            "max": max_value,
            "step": step_value
        }


def input_mission_requirements() -> dict:
    """
    Запрашивает тип цели, тип поражения и требуемую вероятность.
    """
    print("\nТипы целей:")
    print(", ".join(CONFIG["target_types"]))

    while True:
        target_type = input("Тип цели [stronghold]: ").strip()
        target_type = target_type or "stronghold"

        if target_type in CONFIG["target_types"]:
            break

        print("  Ошибка: неизвестный тип цели.")

    print("\nТипы требуемого воздействия:")
    print(", ".join(CONFIG["effect_types"]))

    while True:
        effect_type = input("Тип поражения [destroy]: ").strip()
        effect_type = effect_type or "destroy"

        if effect_type in CONFIG["effect_types"]:
            break

        print("  Ошибка: неизвестный тип воздействия.")

    while True:
        p_required = input(
            "Требуемая вероятность поражения [0.90]: "
        ).strip()

        try:
            p_required = float(p_required) if p_required else 0.90
        except ValueError:
            print("  Ошибка: введите число от 0 до 1.")
            continue

        if 0 < p_required < 1:
            break

        print("  Ошибка: вероятность должна быть в диапазоне от 0 до 1.")

    return {
        "target_type": target_type,
        "effect_type": effect_type,
        "p_required": p_required
    }


print("Модуль application_layer.workflows.user_input_workflow определён.")