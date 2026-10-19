# ЯЧЕЙКА 10. МОДУЛЬ: application_layer/api.py
#
# Назначение:
# - публичные функции для будущего GUI;
# - обучение и сравнение моделей;
# - прямая задача прогнозирования;
# - обратная задача оптимизации;
# - пакетная обработка входного файла.
#
# GUI в дальнейшем будет вызывать только эти функции,
# не обращаясь напрямую к внутренним классам.
# ============================================================
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import joblib
import platform
from sklearn.model_selection import learning_curve

from core_layer.constraint_checker import ConstraintChecker
from core_layer.kill_Probability_Predictor import KillProbabilityPredictor
from core_layer.model_comparator import ModelComparator
from core_layer.model_trainer import ModelTrainer
from core_layer.optimizer import MissionOptimizer
from core_layer.preprocessing import FeaturePreprocessor
from core_layer.visualization import Visualizer
from data_layer.data_loader import DataLoader
from data_layer.data_validator import DataValidator


class ApplicationAPI:
    """
    Публичный программный интерфейс системы.
    """

    def __init__(self, config: dict):
        self.config = config

        self.data_loader = DataLoader()
        self.data_validator = DataValidator(config)
        self.preprocessor = FeaturePreprocessor(config)
        self.trainer = ModelTrainer(config)
        self.comparator = ModelComparator(config)
        self.checker = ConstraintChecker(config)
        self.visualizer = Visualizer() 

        self.trained_models = {}
        self.comparison_report = None
        self.best_model_name = None
        self.predictor = None
        self.optimizer = None
        self.train_test_report = None   
        self.cross_validation_report = None
        self.learning_curves = None
        self.models_dir = Path(self.config["models_dir"])
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.best_model_metadata = None
    # --------------------------------------------------------
    # 1. Подготовка данных
    # --------------------------------------------------------
    def prepare_data(
        self,
        source=None,
        dataframe: pd.DataFrame | None = None,
        n_samples: int = 5000,
        use_sdv: bool = False
    ):
        """
        Загружает реальные данные, принимает готовый DataFrame
        либо генерирует синтетику.
        """
        if dataframe is not None:
            df = dataframe.copy()

        elif source is not None:
            df = self.data_loader.load(source)

        else:
            df = self.data_loader.generate_synthetic(n_samples)

        df = self.data_validator.validate(df)

        if use_sdv:
            df = self.data_loader.generate_with_sdv(
                real_data=df,
                n_samples=n_samples
            )
            df = self.data_validator.validate(df)

        self.raw_data = df
        self.current_dataset_source = source
        return df

    # --------------------------------------------------------
    # 2. Обучение и выбор модели
    # --------------------------------------------------------
    def train_and_select_model(self, tune: bool = False):
        """
        Обучает все модели, сравнивает их и выбирает лучшую.
        """
        X, y = self.preprocessor.fit_transform(self.raw_data)

        X_train, X_test, y_train, y_test = self.preprocessor.split(X, y)

        trained_models = {}
        inference_times = {}


        for model_name in self.config["models"]:
            model, train_time = self.trainer.train(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train
            )


            trained_models[model_name] = model


            # Измеряем время инференса на тестовой выборке
            start_inference = time.perf_counter()
            model.predict(X_test)
            inference_time = time.perf_counter() - start_inference


            inference_times[model_name] = inference_time

            if tune and model_name != "neural_network":
                best_params, best_score = self.trainer.tune_hyperparameters(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    n_trials=self.config["optuna_trials"]
                )

                print(
                    f"{model_name}: лучшие параметры = {best_params}, "
                    f"CV R2 = {best_score:.4f}"
                )

        self.trained_models = trained_models

        self.comparison_report = self.comparator.compare(
            trained_models=trained_models,
            X_test=X_test,
            y_test=y_test,
            inference_times=inference_times
        )

        self.best_model_name = self.comparator.select_best(
            self.comparison_report
        )

        best_model = trained_models[self.best_model_name]

        self.predictor = KillProbabilityPredictor(
            model=best_model,
            preprocessor=self.preprocessor,
            config=self.config
        )

        self.optimizer = MissionOptimizer(
            predictor=self.predictor,
            constraint_checker=self.checker,
            config=self.config
        )

        report_path = self.comparator.save_report(self.comparison_report)
        save_result = self.save_best_model(
            dataset_source=getattr(self, "current_dataset_source", None)
        )
        return {
            "comparison": self.comparison_report,
            "best_model": self.best_model_name,
            "report_path": str(report_path),
            "model_path": save_result["model_path"],
            "metadata_path": save_result["metadata_path"]
        }

    # --------------------------------------------------------
    # 3. Прямая задача
    # --------------------------------------------------------
    def predict_probability(self, mission: dict) -> float:
        """
        Рассчитывает вероятность поражения для одного сценария.
        """
        return self.predictor.predict_single(**mission)

    # --------------------------------------------------------
    # 4. Обратная задача
    # --------------------------------------------------------
    def optimize_mission(
        self,
        target_type: str,
        effect_type: str,
        weather: dict,
        p_required: float,
        grid_step: dict = None,
        fixed_values: dict = None,
        search_ranges: dict = None
    ):
        return self.optimizer.optimize(
            target_type=target_type,
            effect_type=effect_type,
            weather=weather,
            p_required=p_required,
            grid_step=grid_step,
            fixed_values=fixed_values,
            search_ranges=search_ranges
        )

    # --------------------------------------------------------
    # 5. Пакетная обработка
    # --------------------------------------------------------
    def batch_predict(self, input_path: str, output_path: str):
        """
        Выполняет прогноз для всех строк входного файла.

        Входной файл должен содержать признаки,
        но не обязан содержать целевую переменную P_kill.
        """
        if self.predictor is None:
            raise RuntimeError(
                "Модель не обучена. Сначала выполните "
                "prepare_data() и train_and_select_model()."
            )

        df = self.data_loader.load(input_path)

        df = self.data_validator.validate(
            df,
            mode="inference"
        )

        df["P_kill_pred"] = self.predictor.predict(df)

        df = self.checker.check_dataframe(df)

        df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

        return df
        # --------------------------------------------------------
    # 6. Визуализация
    # --------------------------------------------------------
    def plot_model_comparison(self, metric: str = "R2"):
        """
        Строит график сравнения моделей.
        """
        if self.comparison_report is None:
            raise RuntimeError("Сначала выполните train_and_select_model().")

        self.visualizer.plot_model_comparison(
            comparison_df=self.comparison_report,
            metric=metric
        )

    def plot_probability_vs_rockets(
        self,
        target_type: str,
        effect_type: str,
        weather: dict,
        p_required: float,
        H: float,
        D: float,
        V: float
    ):
        """
        Строит график зависимости P_kill от количества ракет.
        """
        N_values = np.arange(
            self.config["constraints"]["N"]["min"],
            self.config["constraints"]["N"]["max"] + 1
        )

        probabilities = []

        for n in N_values:
            p = self.predict_probability({
                "H": H,
                "D": D,
                "V": V,
                "N": int(n),
                "C": target_type,
                "T": effect_type,
                "W_dir": weather["dir"],
                "W_spd": weather["spd"],
                "Temp": weather["temp"]
            })

            probabilities.append(p)

        self.visualizer.plot_probability_vs_rockets(
            N_values=N_values,
            probabilities=probabilities,
            p_required=p_required,
            optimal_n=self._find_min_n(probabilities, p_required)
        )

    @staticmethod
    def _find_min_n(probabilities, p_required):
        """
        Возвращает минимальное N, при котором достигнута требуемая вероятность.
        """
        for n, p in enumerate(probabilities, start=1):
            if p >= p_required:
                return n

        return len(probabilities)
    
    def diagnose_best_model(self):
        """
        Выполняет диагностику лучшей модели:
        train/test-метрики, графики и сохранение отчёта.
        """
        if self.predictor is None:
            raise RuntimeError(
                "Модель не обучена. Сначала выполните "
                "train_and_select_model()."
            )

        X, y = self.preprocessor.fit_transform(self.raw_data)

        X_train, X_test, y_train, y_test = self.preprocessor.split(X, y)

        best_model = self.trained_models[self.best_model_name]

        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        train_metrics = self.comparator.calculate_metrics(
            y_true=y_train,
            y_pred=y_train_pred
        )

        test_metrics = self.comparator.calculate_metrics(
            y_true=y_test,
            y_pred=y_test_pred
        )

        r2_gap = train_metrics["R2"] - test_metrics["R2"]

        self.visualizer.plot_predictions_vs_actual(
            y_true=y_test,
            y_pred=y_test_pred,
            title=(
                f"Фактические и предсказанные значения: "
                f"{self.best_model_name}"
            ),
            filename=f"predictions_vs_actual_{self.best_model_name}.png"
        )

        self.visualizer.plot_residuals(
            y_true=y_test,
            y_pred=y_test_pred,
            title=f"Остатки модели: {self.best_model_name}",
            filename=f"residuals_{self.best_model_name}.png"
        )

        return {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "r2_gap": r2_gap
        }

    # --------------------------------------------------------
    # Train / Test-диагностика всех моделей
    # --------------------------------------------------------
    def diagnose_all_models(self):
        """
        Сравнивает все модели на train и test,
        строит график и сохраняет отчёт.
        """
        if not self.trained_models:
            raise RuntimeError(
                "Модели не обучены. Сначала выполните "
                "train_and_select_model()."
            )

        X, y = self.preprocessor.fit_transform(self.raw_data)

        X_train, X_test, y_train, y_test = self.preprocessor.split(X, y)

        train_test_report = self.comparator.compare_train_test(
            trained_models=self.trained_models,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        self.train_test_report = train_test_report

        report_path = self.comparator.save_train_test_report(
            train_test_df=train_test_report
        )

        self.visualizer.plot_train_test_comparison(
            train_test_df=train_test_report
        )

        return {
            "report": train_test_report,
            "report_path": str(report_path)
        }

    # --------------------------------------------------------
    # Кросс-валидация всех моделей
    # --------------------------------------------------------
    def cross_validate_all_models(self):
        """
        Выполняет кросс-валидацию всех моделей,
        сохраняет отчёт и возвращает результаты.
        """
        if not self.trained_models:
            raise RuntimeError(
                "Модели не обучены. Сначала выполните "
                "train_and_select_model()."
            )

        X, y = self.preprocessor.fit_transform(self.raw_data)

        cv_folds = self.config.get("cv_folds", 5)

        cv_report = self.comparator.cross_validate_models(
            models=self.trained_models,
            X=X,
            y=y,
            cv_folds=cv_folds,
            random_state=self.config.get("random_state", 42),
            exclude_models=[]
        )

        self.cross_validation_report = cv_report

        report_path = self.comparator.save_cross_validation_report(
            cv_df=cv_report
        )

        return {
            "report": cv_report,
            "report_path": str(report_path)
        }

    # --------------------------------------------------------
    # Кривые обучения всех моделей
    # --------------------------------------------------------
    def build_learning_curves(self):
        """
        Строит кривые обучения для моделей,
        совместимых с sklearn.
        """
        if not self.trained_models:
            raise RuntimeError(
                "Модели не обучены. Сначала выполните "
                "train_and_select_model()."
            )

        X, y = self.preprocessor.fit_transform(self.raw_data)

        cv_folds = self.config.get("cv_folds", 5)
        random_state = self.config.get("random_state", 42)

        learning_curves = {}

        exclude_models = []

        for model_name, model in self.trained_models.items():
            if model_name in exclude_models:
                print(
                    f"Модель {model_name} исключена "
                    f"из построения кривых обучения."
                )
                continue

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

            self.visualizer.plot_model_learning_curve(
                model_name=model_name,
                train_sizes=train_sizes,
                train_scores_mean=np.mean(train_scores, axis=1),
                val_scores_mean=np.mean(val_scores, axis=1),
                filename=f"learning_curve_{model_name}.png"
            )

        if not learning_curves:
            raise ValueError(
                "Нет моделей, совместимых с построением кривых обучения."
            )

        self.learning_curves = learning_curves

        self.visualizer.plot_learning_curves(
            learning_curves=learning_curves,
            filename="learning_curves_all_models.png"
        )

        return learning_curves

    def save_best_model(
        self,
        dataset_source: str = None
    ) -> dict:
        """
        Сохраняет лучшую модель, препроцессор и метаданные.

        Возвращает пути к сохранённым файлам.
        """
        if self.predictor is None or self.best_model_name is None:
            raise RuntimeError(
                "Модель не обучена. Сначала выполните "
                "train_and_select_model()."
            )

        model_name = self.best_model_name
        model = self.trained_models[model_name]

        model_path = self.models_dir / "best_model.joblib"
        metadata_path = self.models_dir / "best_model_metadata.json"

        artifact = {
            "model": model,
            "preprocessor": self.preprocessor,
            "model_name": model_name
        }

        joblib.dump(artifact, model_path, compress=3)

        metadata = {
            "model_name": model_name,
            "saved_at": datetime.now().isoformat(),
            "dataset_source": dataset_source,
            "target": self.config["target"],
            "features": self.config["features"],
            "random_state": self.config.get("random_state"),
            "test_size": self.config.get("test_size"),
            "cv_folds": self.config.get("cv_folds"),
            "comparison_report": self.comparison_report.to_dict(
                orient="records"
            ),
            "best_model_metrics": self.comparison_report[
                self.comparison_report["model"] == model_name
            ].to_dict(orient="records")[0],
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__
        }

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=4,
                default=str
            )

        self.best_model_metadata = metadata

        print(f"Лучшая модель сохранена: {model_path}")
        print(f"Метаданные модели сохранены: {metadata_path}")

        return {
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "model_name": model_name
        }

    def load_best_model(
        self,
        model_path: str = None,
        metadata_path: str = None
        ) -> dict:
        """
        Загружает ранее сохранённую лучшую модель
        и восстанавливает состояние API.
        """
        model_path = Path(
            model_path or self.models_dir / "best_model.joblib"
        )

        metadata_path = Path(
            metadata_path or self.models_dir / "best_model_metadata.json"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Файл модели не найден: {model_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Файл метаданных не найден: {metadata_path}"
            )

        artifact = joblib.load(model_path)

        model = artifact["model"]
        preprocessor = artifact["preprocessor"]
        model_name = artifact["model_name"]

        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        self.preprocessor = preprocessor
        self.best_model_name = model_name
        self.best_model_metadata = metadata

        self.trained_models = {
            model_name: model
        }

        self.predictor = KillProbabilityPredictor(
            model=model,
            preprocessor=self.preprocessor,
            config=self.config
        )

        self.optimizer = MissionOptimizer(
            predictor=self.predictor,
            constraint_checker=self.checker,
            config=self.config
        )

        print(f"Загружена модель: {model_name}")
        print(f"Дата сохранения: {metadata.get('saved_at')}")
        print(f"Источник данных: {metadata.get('dataset_source')}")

        return {
            "model_name": model_name,
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "metadata": metadata
        }

print("Модуль application_layer.api определён.")