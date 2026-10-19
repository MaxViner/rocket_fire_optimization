# ============================================================
# МОДУЛЬ: core_layer/model_comparator.py
#
# Назначение:
# - расчёт метрик качества регрессионных моделей;
# - сравнение моделей на train и test;
# - кросс-валидация;
# - формирование и сохранение отчёта.
# ============================================================


import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate


class ModelComparator:
    """
    Сравнивает регрессионные модели и формирует отчёты.
    """

    def __init__(self, config: dict):
        self.config = config

        self.reports_dir = (
            Path(__file__).resolve().parents[1]
            / "storage"
            / "reports"
        )

        self.reports_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Метрики
    # --------------------------------------------------------
    def calculate_metrics(
        self,
        y_true,
        y_pred,
        inference_time_sec: float = None
    ) -> dict:
        """
        Рассчитывает метрики регрессии.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        mse = mean_squared_error(y_true, y_pred)

        metrics = {
            "R2": float(r2_score(y_true, y_pred)),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "MSE": float(mse),
            "RMSE": float(np.sqrt(mse))
        }

        if inference_time_sec is not None:
            metrics["inference_time_sec"] = float(inference_time_sec)

        return metrics

    # --------------------------------------------------------
    # Сравнение моделей на test
    # --------------------------------------------------------
    def compare(
        self,
        trained_models: dict,
        X_test,
        y_test,
        inference_times: dict = None
    ) -> pd.DataFrame:
        inference_times = inference_times or {}

        results = []

        for model_name, model in trained_models.items():
            y_pred = model.predict(X_test)

            metrics = self.calculate_metrics(
                y_true=y_test,
                y_pred=y_pred,
                inference_time_sec=inference_times.get(model_name)
            )

            metrics["model"] = model_name
            results.append(metrics)

        comparison = pd.DataFrame(results)

        columns = [
            "model",
            "R2",
            "MAE",
            "MSE",
            "RMSE"
        ]

        if "inference_time_sec" in comparison.columns:
            columns.append("inference_time_sec")

        return comparison[columns].sort_values(
            by="R2",
            ascending=False
        ).reset_index(drop=True)

    # --------------------------------------------------------
    # Train / Test диагностика
    # --------------------------------------------------------
    def compare_train_test(
        self,
        trained_models: dict,
        X_train,
        y_train,
        X_test,
        y_test
    ) -> pd.DataFrame:
        """
        Сравнивает качество моделей на train и test.

        Большая разница R2 между train и test
        может указывать на переобучение.
        """
        results = []

        for model_name, model in trained_models.items():
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_metrics = self.calculate_metrics(
                y_true=y_train,
                y_pred=y_train_pred
            )

            test_metrics = self.calculate_metrics(
                y_true=y_test,
                y_pred=y_test_pred
            )

            results.append({
                "model": model_name,
                "train_R2": train_metrics["R2"],
                "test_R2": test_metrics["R2"],
                "R2_gap": train_metrics["R2"] - test_metrics["R2"],
                "train_RMSE": train_metrics["RMSE"],
                "test_RMSE": test_metrics["RMSE"],
                "RMSE_gap": test_metrics["RMSE"] - train_metrics["RMSE"],
                "train_MAE": train_metrics["MAE"],
                "test_MAE": test_metrics["MAE"]
            })

        return pd.DataFrame(results).sort_values(
            by="test_R2",
            ascending=False
        ).reset_index(drop=True)

    # --------------------------------------------------------
    # Кросс-валидация
    # --------------------------------------------------------
    def cross_validate_models(
        self,
        models: dict,
        X,
        y,
        cv_folds: int = 5,
        random_state: int = 42,
        exclude_models: list[str] = None
    ) -> pd.DataFrame:
        """
        Выполняет k-fold кросс-валидацию
        для совместимых с sklearn моделей.
        """
        exclude_models = exclude_models or []

        cv = KFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state
        )

        scoring = {
            "R2": "r2",
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error"
        }

        results = []

        for model_name, model in models.items():
            if model_name in exclude_models:
                # print(
                #     f"Модель {model_name} исключена "
                #     f"из кросс-валидации."
                # )
                continue

            cv_result = cross_validate(
                estimator=model,
                X=X,
                y=y,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1
            )

            results.append({
                "model": model_name,
                "cv_folds": cv_folds,

                "train_R2_mean": float(np.mean(cv_result["train_R2"])),
                "train_R2_std": float(np.std(cv_result["train_R2"])),

                "val_R2_mean": float(np.mean(cv_result["test_R2"])),
                "val_R2_std": float(np.std(cv_result["test_R2"])),

                "R2_gap_mean": float(
                    np.mean(cv_result["train_R2"])
                    - np.mean(cv_result["test_R2"])
                ),

                "train_RMSE_mean": float(
                    -np.mean(cv_result["train_RMSE"])
                ),
                "val_RMSE_mean": float(
                    -np.mean(cv_result["test_RMSE"])
                ),

                "train_MAE_mean": float(
                    -np.mean(cv_result["train_MAE"])
                ),
                "val_MAE_mean": float(
                    -np.mean(cv_result["test_MAE"])
                )
            })

        if not results:
            raise ValueError(
                "Нет моделей, совместимых с кросс-валидацией."
            )

        return pd.DataFrame(results).sort_values(
            by="val_R2_mean",
            ascending=False
        ).reset_index(drop=True)

    # --------------------------------------------------------
    # Выбор лучшей модели
    # --------------------------------------------------------
    def select_best(self, comparison_df: pd.DataFrame) -> str:
        """
        Выбирает модель с максимальным R2 на тестовой выборке.
        """
        if comparison_df is None or comparison_df.empty:
            raise ValueError("Нет данных для выбора лучшей модели.")

        best_row = comparison_df.sort_values(
            by="R2",
            ascending=False
        ).iloc[0]

        return best_row["model"]

    # --------------------------------------------------------
    # Сохранение отчётов
    # --------------------------------------------------------
    def save_report(
        self,
        comparison_df: pd.DataFrame,
        filename: str = "model_comparison_report"
    ) -> Path:
        """
        Сохраняет отчёт сравнения моделей в CSV и JSON.
        """
        csv_path = self.reports_dir / f"{filename}.csv"
        json_path = self.reports_dir / f"{filename}.json"

        comparison_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(
                comparison_df.to_dict(orient="records"),
                file,
                ensure_ascii=False,
                indent=4
            )

        return csv_path

    def save_train_test_report(
        self,
        train_test_df: pd.DataFrame,
        filename: str = "train_test_diagnostic_report"
    ) -> Path:
        """
        Сохраняет отчёт train/test-диагностики.
        """
        csv_path = self.reports_dir / f"{filename}.csv"
        json_path = self.reports_dir / f"{filename}.json"

        train_test_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(
                train_test_df.to_dict(orient="records"),
                file,
                ensure_ascii=False,
                indent=4
            )

        return csv_path

    def save_cross_validation_report(
        self,
        cv_df: pd.DataFrame,
        filename: str = "cross_validation_report"
    ) -> Path:
        """
        Сохраняет отчёт кросс-валидации.
        """
        csv_path = self.reports_dir / f"{filename}.csv"
        json_path = self.reports_dir / f"{filename}.json"

        cv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(
                cv_df.to_dict(orient="records"),
                file,
                ensure_ascii=False,
                indent=4
            )

        return csv_path


print("Модуль core_layer.model_comparator определён.")