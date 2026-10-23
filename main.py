# ============================================================
# ФАЙЛ: main.py
#
# Точка входа в систему.
# Формирует отдельный отчёт для каждого запуска.
# ============================================================


import pandas as pd

from application_layer.api import ApplicationAPI
from application_layer.workflows.batch_workflow import run_batch_processing
from application_layer.workflows.dataset_workflow import choose_dataset
from application_layer.workflows.optimization_workflow import (
    plot_optimization_result,
    print_optimization_result,
    run_optimization,
)
from application_layer.workflows.user_input_workflow import choose_conditions_mode
from config import CONFIG, create_directories


def choose_model_mode() -> int:
    """
    Выбирает режим работы с прогнозирующей моделью.
    """
    print("\nВыберите режим работы с моделью:")
    print("1 — Обучить модели заново и выбрать лучшую")
    print("2 — Загрузить ранее сохранённую лучшую модель")

    while True:
        choice = input("Введите номер режима (1 или 2): ").strip()

        if choice in ("1", "2"):
            return int(choice)

        print("Ошибка: введите 1 или 2.")


def train_models_and_add_to_report(api: ApplicationAPI) -> dict:
    """
    Обучает модели, выбирает лучшую
    и добавляет результаты в отчёт.
    """
    print("\n[2/6] Обучение и сравнение моделей...")

    training_result = api.train_and_select_model(tune=False)

    comparison = training_result["comparison"]

    print(f"Лучшая модель: {training_result['best_model']}")
    print("\nМетрики моделей:")
    print(comparison.to_string(index=False))

    api.report_service.add_table(
        title="Результаты обучения и сравнения моделей",
        dataframe=comparison,
        filename="model_comparison.csv",
        note=(
            "В качестве основной модели выбрана модель с наилучшим "
            "значением коэффициента детерминации R²."
        )
    )

    return training_result


def load_saved_model_and_add_to_report(api: ApplicationAPI) -> dict:
    """
    Загружает ранее сохранённую модель
    и добавляет сведения в отчёт.
    """
    print("\n[2/6] Загрузка ранее сохранённой модели...")

    load_result = api.load_best_model()

    metadata = load_result["metadata"]

    print(f"Загружена модель: {load_result['model_name']}")
    print(f"Дата сохранения: {metadata.get('saved_at')}")
    print(f"Источник данных: {metadata.get('dataset_source')}")

    comparison_df = pd.DataFrame(
        metadata.get("comparison_report", [])
    )

    if not comparison_df.empty:
        api.report_service.add_table(
            title="Метрики ранее обученных моделей",
            dataframe=comparison_df,
            filename="saved_model_comparison.csv",
            note=(
                "Использована ранее сохранённая лучшая модель. "
                "Повторное обучение моделей не выполнялось."
            )
        )

    api.report_service.add_section(
        title="Загруженная прогнозирующая модель",
        content=(
            f"**Тип модели:** {load_result['model_name']}  \n"
            f"**Дата сохранения:** {metadata.get('saved_at')}  \n"
            f"**Источник обучающих данных:** "
            f"{metadata.get('dataset_source')}  \n"
            f"**Файл модели:** `{load_result['model_path']}`."
        )
    )

    return load_result


def run_model_diagnostics(api: ApplicationAPI):
    """
    Выполняет диагностику качества обученных моделей
    и добавляет результаты в отчёт текущего запуска.
    """
    print("\n[Дополнительно] Train/test-диагностика моделей...")

    diagnostics = api.diagnose_all_models()

    train_test_report = diagnostics["report"]

    print("\nTrain/test-отчёт:")
    print(train_test_report.to_string(index=False))

    api.report_service.add_table(
        title="Сравнение качества моделей на обучающей и тестовой выборках",
        dataframe=train_test_report,
        filename="train_test_report.csv",
        note=(
            "Малый разрыв между метриками на обучающей и тестовой выборках "
            "свидетельствует об отсутствии выраженного переобучения."
        )
    )

    print("\n[Дополнительно] Кросс-валидация моделей...")

    cv_result = api.cross_validate_all_models()
    cv_report = cv_result["report"]

    print("\nРезультаты кросс-валидации:")
    print(cv_report.to_string(index=False))

    api.report_service.add_table(
        title="Результаты кросс-валидации моделей",
        dataframe=cv_report,
        filename="cross_validation_report.csv",
        note=(
            "Кросс-валидация позволяет оценить устойчивость качества "
            "моделей при изменении состава обучающих данных."
        )
    )

    print("\n[Дополнительно] Построение кривых обучения...")

    learning_curves = api.build_learning_curves()

    print("\nПостроены кривые обучения для моделей:")
    print(", ".join(learning_curves.keys()))

    for model_name in learning_curves.keys():
        api.report_service.add_figure(
            title=f"Кривая обучения: {model_name}",
            filename=f"krivaya_obucheniya_{model_name}.png",
            description=(
                "Зависимость качества модели от объёма обучающей выборки."
            )
        )

    print("\n[Дополнительно] Диагностика лучшей модели...")

    best_diagnostics = api.diagnose_best_model()

    print("\nМетрики лучшей модели на train:")
    print(best_diagnostics["train_metrics"])

    print("\nМетрики лучшей модели на test:")
    print(best_diagnostics["test_metrics"])

    print(
        f"\nРазница R2 (train - test): "
        f"{best_diagnostics['r2_gap']:.4f}"
    )

    api.report_service.add_figure(
        title="Фактические и прогнозные значения вероятности поражения",
        filename=f"fakt_i_prognoz_{api.best_model_name}.png",
        description=(
            "Сопоставление фактических и прогнозных значений "
            "вероятности поражения для лучшей модели."
        )
    )

    api.report_service.add_figure(
        title="Анализ остатков лучшей модели",
        filename=f"ostatki_{api.best_model_name}.png",
        description=(
            "Распределение остатков позволяет оценить систематические "
            "ошибки прогнозирования лучшей модели."
        )
    )


def main():
    """
    Точка входа в приложение.
    """
    print("=" * 60)
    print("СИСТЕМА ПРОГНОЗИРОВАНИЯ И ОПТИМИЗАЦИИ")
    print("=" * 60)

    create_directories()

    api = ApplicationAPI(CONFIG)

    run_dir = api.report_service.start_run()

    print(f"Каталог текущего отчёта: {run_dir}")

    # --------------------------------------------------------
    # 1. Подготовка данных
    # --------------------------------------------------------
    print("\n[1/6] Подготовка данных...")

    df, source = choose_dataset()

    df = api.prepare_data(
        source=source,
        dataframe=df,
        n_samples=5000,
        use_sdv=False
    )

    print(
        f"Датасет подготовлен: "
        f"{df.shape[0]} строк, {df.shape[1]} столбцов."
    )

    if source:
        print(f"Источник данных: {source}")

    api.report_service.add_section(
        title="Исходные данные",
        content=(
            f"Количество записей: **{df.shape[0]}**.  \n"
            f"Количество признаков и целевой переменной: **{df.shape[1]}**.  \n"
            f"Источник данных: **{source or 'синтетическая генерация'}**."
        )
    )

    # --------------------------------------------------------
    # 2. Обучение или загрузка модели
    # --------------------------------------------------------
    model_mode = choose_model_mode()

    if model_mode == 1:
        training_result = train_models_and_add_to_report(api)

        best_model_name = training_result["best_model"]

        # ----------------------------------------------------
        # 3. Диагностика и визуализация
        # ----------------------------------------------------
        print("\n[3/6] Диагностика и визуализация моделей...")

        run_model_diagnostics(api)

        api.plot_model_comparison(metric="R2")
        api.report_service.add_figure(
            title="Сравнение моделей по коэффициенту детерминации",
            filename="sravnenie_modeley_R2.png",
            description=(
                "Коэффициент детерминации характеризует долю объяснённой "
                "дисперсии вероятности поражения."
            )
        )

        api.plot_model_comparison(metric="RMSE")
        api.report_service.add_figure(
            title="Сравнение моделей по среднеквадратической ошибке",
            filename="sravnenie_modeley_RMSE.png",
            description=(
                "Среднеквадратическая ошибка усиливает влияние крупных "
                "ошибок прогнозирования."
            )
        )

        api.plot_model_comparison(metric="inference_time_sec")
        api.report_service.add_figure(
            title="Сравнение моделей по времени формирования прогноза",
            filename="sravnenie_modeley_vremya.png",
            description=(
                "Время формирования прогноза характеризует вычислительную "
                "эффективность моделей."
            )
        )

    else:
        load_result = load_saved_model_and_add_to_report(api)

        best_model_name = load_result["model_name"]

        print(
            "\n[3/6] Диагностика пропущена: "
            "используется ранее обученная модель."
        )

        api.report_service.add_section(
            title="Диагностика моделей",
            content=(
                "Диагностика и повторное обучение моделей не выполнялись, "
                "так как используется ранее сохранённая лучшая модель."
            )
        )

    # --------------------------------------------------------
    # 4. Выбор условий боевого применения
    # --------------------------------------------------------
    print("\n[4/6] Задание условий боевого применения...")

    application_conditions, weather, mission_requirements = (
        choose_conditions_mode()
    )
    target_type = mission_requirements.get(
        "target_type",
        "не задан"
    )

    effect_type = mission_requirements.get(
        "effect_type",
        "не задан"
    )

    p_required = mission_requirements.get(
        "p_required",
        "не задана"
    )

    api.report_service.add_section(
        title="Условия боевого применения",
        content=(
            f"**Тип цели:** {target_type}  \n"
            f"**Требуемый тип воздействия:** {effect_type}  \n"
            f"**Требуемая вероятность поражения:** {p_required}  \n"
            f"**Направление ветра:** {weather['dir']} град.  \n"
            f"**Скорость ветра:** {weather['spd']} м/с  \n"
            f"**Температура воздуха:** {weather['temp']} °C."
        )
    )

    # --------------------------------------------------------
    # 5. Обратная задача
    # --------------------------------------------------------
    print("\n[5/6] Обратная задача...")

    optimization_result = run_optimization(
        api=api,
        application_conditions=application_conditions,
        weather=weather,
        mission_requirements=mission_requirements
    )

    best = print_optimization_result(optimization_result)

    if best is None:
        api.report_service.add_section(
            title="Результаты решения обратной задачи",
            content=(
                "В заданной области допустимых параметров не найдено "
                "решения, обеспечивающего требуемую вероятность поражения."
            )
        )

        api.report_service.save_report(
            run_info={
                "Лучший прогнозирующий алгоритм": api.best_model_name,
                "Источник данных": source or "синтетическая генерация"
            }
        )

        return

    api.report_service.add_optimization_result(
        result=optimization_result,
        best=best
    )

    plot_optimization_result(
        api=api,
        best=best,
        weather=weather,
        mission_requirements=mission_requirements
    )

    api.report_service.add_figure(
        title="Зависимость вероятности поражения от количества НАР",
        filename="zavisimost_P_ot_N.png",
        description=(
            "График показывает минимальное количество НАР с кассетной "
            "боевой частью, обеспечивающее требуемую вероятность поражения."
        )
    )

    # --------------------------------------------------------
    # 6. Пакетная обработка
    # --------------------------------------------------------
    print("\n[6/6] Пакетная обработка...")

    batch_result = run_batch_processing(api)

    if batch_result is not None:
        api.report_service.add_section(
            title="Пакетная обработка сценариев",
            content=(
                "Выполнена пакетная обработка входных сценариев. "
                "Результаты сохранены в соответствующем выходном файле."
            )
        )

    api.report_service.save_report(
        run_info={
            "Лучший прогнозирующий алгоритм": api.best_model_name,
            "Источник данных": source or "синтетическая генерация",
            "Количество записей в выборке": df.shape[0],
            "Режим работы с моделью": (
                "новое обучение" if model_mode == 1
                else "заг1рузка сохранённой модели"
            )
        }
    )

    print("\n" + "=" * 60)
    print("ВЫПОЛНЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)


if __name__ == "__main__":
    main()