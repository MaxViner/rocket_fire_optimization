# ============================================================
# ФАЙЛ: main.py
#
# Точка входа в систему.

# запускает сценарий и вызывает функции
# из application_layer.workflows.
# ============================================================


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


def run_model_diagnostics(api: ApplicationAPI):
    """
    Выполняет диагностику качества обученных моделей.
    """
    print("\n[Дополнительно] Train/test-диагностика моделей...")

    diagnostics = api.diagnose_all_models()

    print("\nTrain/test-отчёт:")
    print(diagnostics["report"].to_string(index=False))

    print("\n[Дополнительно] Кросс-валидация моделей...")

    cv_result = api.cross_validate_all_models()

    print("\nРезультаты кросс-валидации:")
    print(cv_result["report"].to_string(index=False))

    print("\n[Дополнительно] Построение кривых обучения...")

    learning_curves = api.build_learning_curves()

    print("\nПостроены кривые обучения для моделей:")
    print(", ".join(learning_curves.keys()))

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


def main():
    """
    Точка входа в приложение.
    """
    print("=" * 60)
    print("СИСТЕМА ПРОГНОЗИРОВАНИЯ И ОПТИМИЗАЦИИ")
    print("=" * 60)

    create_directories()

    api = ApplicationAPI(CONFIG)

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

    print(f"Датасет подготовлен: {df.shape[0]} строк, {df.shape[1]} столбцов.")

    if source:
        print(f"Источник данных: {source}")

    # --------------------------------------------------------
    # 2. Обучение моделей
    # --------------------------------------------------------
    print("\n[2/6] Обучение и сравнение моделей...")

    training_result = api.train_and_select_model(tune=False)

    print(f"Лучшая модель: {training_result['best_model']}")
    print("\nМетрики моделей:")
    print(training_result["comparison"].to_string(index=False))

    # --------------------------------------------------------
    # 3. Диагностика и визуализация
    # --------------------------------------------------------
    print("\n[3/6] Диагностика и визуализация моделей...")

    run_model_diagnostics(api)

    api.plot_model_comparison(metric="R2")
    api.plot_model_comparison(metric="RMSE")
    api.plot_model_comparison(metric="inference_time_sec")

    # --------------------------------------------------------
    # 4. Выбор условий боевого применения
    # --------------------------------------------------------
    print("\n[4/6] Задание условий боевого применения...")

    application_conditions, weather, mission_requirements = (
        choose_conditions_mode()
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
        return

    plot_optimization_result(
        api=api,
        best=best,
        weather=weather,
        mission_requirements=mission_requirements
    )

    # --------------------------------------------------------
    # 6. Пакетная обработка
    # --------------------------------------------------------
    run_batch_processing(api)

    print("\n" + "=" * 60)
    print("ВЫПОЛНЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)


if __name__ == "__main__":
    main()