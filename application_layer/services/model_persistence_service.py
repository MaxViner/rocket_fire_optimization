# ============================================================
# МОДУЛЬ: application_layer/services/model_persistence_service.py
#
# Назначение:
# - сохранение лучшей модели и препроцессора;
# - сохранение метаданных обучения;
# - загрузка ранее обученной модели.
# ============================================================


import json
import platform
from datetime import datetime
from pathlib import Path

import joblib
import sklearn

from core_layer.killProbabilityPredictor import KillProbabilityPredictor
from core_layer.optimizer import MissionOptimizer


class ModelPersistenceService:
    """
    Отвечает за сохранение и загрузку обученной модели.
    """

    def __init__(self, config: dict):
        self.config = config

        self.models_dir = Path(config["models_dir"])
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_best_model(
        self,
        api,
        dataset_source: str = None
    ) -> dict:
        """
        Сохраняет лучшую модель, препроцессор и метаданные.
        """
        if api.predictor is None or api.best_model_name is None:
            raise RuntimeError(
                "Модель не обучена. Сначала выполните "
                "train_and_select_model()."
            )

        model_name = api.best_model_name
        model = api.trained_models[model_name]

        model_path = self.models_dir / "best_model.joblib"
        metadata_path = self.models_dir / "best_model_metadata.json"

        artifact = {
            "model": model,
            "preprocessor": api.preprocessor,
            "model_name": model_name
        }

        joblib.dump(artifact, model_path, compress=3)

        metadata = self._build_metadata(
            api=api,
            model_name=model_name,
            dataset_source=dataset_source
        )

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=4,
                default=str
            )

        api.best_model_metadata = metadata

        print(f"Лучшая модель сохранена: {model_path}")
        print(f"Метаданные модели сохранены: {metadata_path}")

        return {
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "model_name": model_name
        }

    def load_best_model(
        self,
        api,
        model_path: str = None,
        metadata_path: str = None
    ) -> dict:
        """
        Загружает ранее сохранённую модель
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

        api.preprocessor = preprocessor
        api.best_model_name = model_name
        api.best_model_metadata = metadata
        api.trained_models = {
            model_name: model
        }

        api.predictor = KillProbabilityPredictor(
            model=model,
            preprocessor=api.preprocessor,
            config=self.config
        )

        api.optimizer = MissionOptimizer(
            predictor=api.predictor,
            constraint_checker=api.checker,
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

    def _build_metadata(
        self,
        api,
        model_name: str,
        dataset_source: str
    ) -> dict:
        """
        Формирует метаданные обученной модели.
        """
        best_metrics = api.comparison_report[
            api.comparison_report["model"] == model_name
        ].to_dict(orient="records")

        return {
            "model_name": model_name,
            "saved_at": datetime.now().isoformat(),
            "dataset_source": dataset_source,
            "target": self.config["target"],
            "features": self.config["features"],
            "random_state": self.config.get("random_state"),
            "test_size": self.config.get("test_size"),
            "cv_folds": self.config.get("cv_folds"),
            "comparison_report": api.comparison_report.to_dict(
                orient="records"
            ),
            "best_model_metrics": best_metrics,
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__
        }


print("Модуль application_layer.services.model_persistence_service определён.")