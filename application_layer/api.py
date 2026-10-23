# ============================================================
# МОДУЛЬ: application_layer/api.py
#
# Назначение:
# - публичный программный интерфейс системы;
# - тонкий слой доступа к сервисам и workflow.
# ============================================================


import numpy as np
import pandas as pd

from application_layer.services.diagnostics_service import DiagnosticsService
from application_layer.services.model_persistence_service import ModelPersistenceService
from application_layer.services.report_service import ReportService
from core_layer.constraint_checker import ConstraintChecker
from core_layer.killProbabilityPredictor import KillProbabilityPredictor
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

        self.persistence_service = ModelPersistenceService(config)
        self.diagnostics_service = DiagnosticsService(config)

        self.trained_models = {}
        self.comparison_report = None
        self.best_model_name = None
        self.predictor = None
        self.optimizer = None

        self.train_test_report = None
        self.cross_validation_report = None
        self.learning_curves = None
        self.best_model_metadata = None
        self.current_dataset_source = None
        self.report_service = ReportService(config)
    # --------------------------------------------------------
    # Данные
    # --------------------------------------------------------
    def prepare_data(
        self,
        source=None,
        dataframe: pd.DataFrame | None = None,
        n_samples: int = 5000,
        use_sdv: bool = False
    ):
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
    # Обучение
    # --------------------------------------------------------
    def train_and_select_model(self, tune: bool = False):
        X, y = self.preprocessor.fit_transform(self.raw_data)

        X_train, X_test, y_train, y_test = self.preprocessor.split(X, y)

        trained_models = {}
        inference_times = {}

        for model_name in self.config["models"]:
            model, _ = self.trainer.train(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train
            )

            trained_models[model_name] = model

            start = __import__("time").perf_counter()
            model.predict(X_test)
            inference_times[model_name] = (
                __import__("time").perf_counter() - start
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

        report_path = self.comparator.save_report(
            self.comparison_report
        )

        save_result = self.persistence_service.save_best_model(
            api=self,
            dataset_source=self.current_dataset_source
        )

        return {
            "comparison": self.comparison_report,
            "best_model": self.best_model_name,
            "report_path": str(report_path),
            "model_path": save_result["model_path"],
            "metadata_path": save_result["metadata_path"]
        }

    # --------------------------------------------------------
    # Прогноз и оптимизация
    # --------------------------------------------------------
    def predict_probability(self, mission: dict) -> float:
        return self.predictor.predict_single(**mission)

    def optimize_mission(
        self,
        target_type: str,
        effect_type: str,
        weather: dict,
        p_required: float,
        search_ranges: dict = None,
        coarse_factor: int = 4,
        show_progress: bool = True
    ):
        """
        Решает обратную задачу оптимизации параметров пуска.
        """
        return self.optimizer.optimize(
            target_type=target_type,
            effect_type=effect_type,
            weather=weather,
            p_required=p_required,
            search_ranges=search_ranges,
            coarse_factor=coarse_factor,
            show_progress=show_progress
        )

    # --------------------------------------------------------
    # Пакетная обработка
    # --------------------------------------------------------
    def batch_predict(self, input_path: str, output_path: str):
        if self.predictor is None:
            raise RuntimeError(
                "Модель не обучена или не загружена."
            )

        df = self.data_loader.load(input_path)
        df = self.data_validator.validate(df, mode="inference")

        df["P_kill_pred"] = self.predictor.predict(df)
        df = self.checker.check_dataframe(df)

        df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

        return df

    # --------------------------------------------------------
    # Визуализация
    # --------------------------------------------------------
    def plot_model_comparison(self, metric: str = "R2"):
        if self.comparison_report is None:
            raise RuntimeError(
                "Сначала выполните train_and_select_model()."
            )

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
            optimal_n=self._find_min_n(
                probabilities,
                p_required
            )
        )

    @staticmethod
    def _find_min_n(probabilities, p_required):
        for n, p in enumerate(probabilities, start=1):
            if p >= p_required:
                return n

        return len(probabilities)

    # --------------------------------------------------------
    # Прокси к сервисам
    # --------------------------------------------------------
    def save_best_model(self, dataset_source: str = None):
        return self.persistence_service.save_best_model(
            api=self,
            dataset_source=dataset_source
        )

    def load_best_model(
        self,
        model_path: str = None,
        metadata_path: str = None
    ):
        return self.persistence_service.load_best_model(
            api=self,
            model_path=model_path,
            metadata_path=metadata_path
        )

    def diagnose_best_model(self):
        return self.diagnostics_service.diagnose_best_model(self)

    def diagnose_all_models(self):
        return self.diagnostics_service.diagnose_all_models(self)

    def cross_validate_all_models(self):
        return self.diagnostics_service.cross_validate_all_models(self)

    def build_learning_curves(self):
        return self.diagnostics_service.build_learning_curves(self)


print("Модуль application_layer.api определён.")