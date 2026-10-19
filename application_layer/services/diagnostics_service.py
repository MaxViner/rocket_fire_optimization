# ============================================================
# МОДУЛЬ: application_layer/services/diagnostics_service.py
#
# Назначение:
# - диагностика лучшей модели;
# - train/test-сравнение всех моделей;
# - кросс-валидация;
# - построение кривых обучения.
# ============================================================


import numpy as np
from sklearn.model_selection import learning_curve

from core_layer.killProbabilityPredictor import KillProbabilityPredictor
from core_layer.optimizer import MissionOptimizer


class DiagnosticsService:
    """
    Отвечает за диагностику качества обученных моделей.
    """

    def __init__(self, config: dict):
        self.config = config

    def diagnose_best_model(self, api) -> dict:
        """
        Выполняет диагностику лучшей модели.
        """
        if api.predictor is None:
            raise RuntimeError(
                "Модель не обучена. Сначала выполните "
                "train_and_select_model()."
            )

        X, y = api.preprocessor.fit_transform(api.raw_data)

        X_train, X_test, y_train, y_test = api.preprocessor.split(X, y)

        best_model = api.trained_models[api.best_model_name]

        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        train_metrics = api.comparator.calculate_metrics(
            y_true=y_train,
            y_pred=y_train_pred
        )

        test_metrics = api.comparator.calculate_metrics(
            y_true=y_test,
            y_pred=y_test_pred
        )

        r2_gap = train_metrics["R2"] - test_metrics["R2"]

        api.visualizer.plot_predictions_vs_actual(
            y_true=y_test,
            y_pred=y_test_pred,
            title=(
                f"Фактические и прогнозные значения: "
                f"{api.best_model_name}"
            ),
            filename=(
                f"fakt_i_prognoz_{api.best_model_name}.png"
            )
        )

        api.visualizer.plot_residuals(
            y_true=y_test,
            y_pred=y_test_pred,
            title=f"Анализ остатков модели: {api.best_model_name}",
            filename=f"ostatki_{api.best_model_name}.png"
        )

        return {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "r2_gap": r2_gap
        }

    def diagnose_all_models(self, api) -> dict:
        """
        Сравнивает все модели на train и test.
        """
        if not api.trained_models:
            raise RuntimeError(
                "Модели не обучены. Сначала выполните "
                "train_and_select_model()."
            )

        X, y = api.preprocessor.fit_transform(api.raw_data)

        X_train, X_test, y_train, y_test = api.preprocessor.split(X, y)

        report = api.comparator.compare_train_test(
            trained_models=api.trained_models,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        api.train_test_report = report

        report_path = api.comparator.save_train_test_report(
            train_test_df=report
        )

        api.visualizer.plot_train_test_comparison(
            train_test_df=report
        )

        return {
            "report": report,
            "report_path": str(report_path)
        }

    def cross_validate_all_models(self, api) -> dict:
        """
        Выполняет кросс-валидацию всех моделей.
        """
        if not api.trained_models:
            raise RuntimeError(
                "Модели не обучены. Сначала выполните "
                "train_and_select_model()."
            )

        X, y = api.preprocessor.fit_transform(api.raw_data)

        cv_report = api.comparator.cross_validate_models(
            models=api.trained_models,
            X=X,
            y=y,
            cv_folds=self.config.get("cv_folds", 5),
            random_state=self.config.get("random_state", 42),
            exclude_models=[]
        )

        api.cross_validation_report = cv_report

        report_path = api.comparator.save_cross_validation_report(
            cv_df=cv_report
        )

        return {
            "report": cv_report,
            "report_path": str(report_path)
        }

    def build_learning_curves(self, api) -> dict:
        """
        Строит кривые обучения всех моделей.
        """
        if not api.trained_models:
            raise RuntimeError(
                "Модели не обучены. Сначала выполните "
                "train_and_select_model()."
            )

        X, y = api.preprocessor.fit_transform(api.raw_data)

        cv_folds = self.config.get("cv_folds", 5)
        random_state = self.config.get("random_state", 42)

        learning_curves = {}

        for model_name, model in api.trained_models.items():
            train_sizes, train_scores, val_scores = learning_curve(
                estimator=model,
                X=X,
                y=y,
                cv=cv_folds,
                scoring="r2",
                train_sizes=np.linspace(0.2, 1.0, 5),
                n_jobs=-1,
                shuffle=True,
                random_state=random_state
            )

            learning_curves[model_name] = {
                "train_sizes": train_sizes.tolist(),
                "train_scores_mean": np.mean(train_scores, axis=1).tolist(),
                "train_scores_std": np.std(train_scores, axis=1).tolist(),
                "val_scores_mean": np.mean(val_scores, axis=1).tolist(),
                "val_scores_std": np.std(val_scores, axis=1).tolist()
            }

            api.visualizer.plot_model_learning_curve(
                model_name=model_name,
                train_sizes=train_sizes,
                train_scores_mean=np.mean(train_scores, axis=1),
                val_scores_mean=np.mean(val_scores, axis=1),
                filename=f"krivaya_obucheniya_{model_name}.png"
            )

        api.learning_curves = learning_curves

        api.visualizer.plot_learning_curves(
            learning_curves=learning_curves,
            filename="krivye_obucheniya.png"
        )

        return learning_curves


print("Модуль application_layer.services.diagnostics_service определён.")