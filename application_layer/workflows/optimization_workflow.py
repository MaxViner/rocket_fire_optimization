# ============================================================
# МОДУЛЬ: application_layer/workflows/optimization_workflow.py
#
# Назначение:
# - обратная задача оптимизации;
# - вывод оптимального решения;
# - построение графика P_kill(N).
# ============================================================


from application_layer.api import ApplicationAPI


def run_optimization(
    api: ApplicationAPI,
    application_conditions: dict,
    weather: dict,
    mission_requirements: dict
) -> dict:
    """
    Выполняет обратную задачу оптимизации.
    """
    print("\nПоиск оптимальных условий пуска...")

    search_ranges = {
        "H": {
            "min": application_conditions["H"]["min"],
            "max": application_conditions["H"]["max"],
            "step": application_conditions["H"]["step"]
        },
        "D": {
            "min": application_conditions["D"]["min"],
            "max": application_conditions["D"]["max"],
            "step": application_conditions["D"]["step"]
        },
        "V": {
            "min": application_conditions["V"]["min"],
            "max": application_conditions["V"]["max"],
            "step": application_conditions["V"]["step"]
        },
        "N": {
            "min": 1,
            "max": 20,
            "step": 1
        }
    }

    return api.optimize_mission(
        target_type=mission_requirements["target_type"],
        effect_type=mission_requirements["effect_type"],
        weather=weather,
        p_required=mission_requirements["p_required"],
        search_ranges=search_ranges,
        coarse_factor=4,
        show_progress=True
    )

def print_optimization_result(optimization_result: dict) -> dict | None:
    """
    Выводит результат оптимизации.
    """
    if not optimization_result["found"]:
        print(
            optimization_result.get(
                "message",
                "Допустимое решение не найдено."
            )
        )
        return None

    best = optimization_result["best"]

    print("\nОптимальные условия пуска:")
    print(f"  Минимальное количество ракет N = {int(best['N'])}")
    print(f"  Высота пуска H = {best['H']:.1f} м")
    print(f"  Дальность до цели D = {best['D']:.1f} м")
    print(f"  Скорость V = {best['V']:.1f} м/с")
    print(f"  Расчётная вероятность поражения P_kill = {best['P_kill']:.4f}")
    print(f"  Требуемая вероятность P_required = {best['P_required']:.2f}")

    print("\nАльтернативные допустимые варианты:")
    print(
        optimization_result["solutions"]
        .head(10)
        .to_string(index=False)
    )

    return best


def plot_optimization_result(
    api: ApplicationAPI,
    best: dict,
    weather: dict,
    mission_requirements: dict
):
    """
    Строит график зависимости P_kill от количества ракет.
    """
    print("\nПостроение зависимости P_kill от количества ракет...")

    api.plot_probability_vs_rockets(
        target_type=mission_requirements["target_type"],
        effect_type=mission_requirements["effect_type"],
        weather=weather,
        p_required=mission_requirements["p_required"],
        H=best["H"],
        D=best["D"],
        V=best["V"]
    )


print("Модуль application_layer.workflows.optimization_workflow определён.")