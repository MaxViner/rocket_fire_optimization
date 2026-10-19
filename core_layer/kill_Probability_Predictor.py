# ============================================================
# ЯЧЕЙКА 7. МОДУЛЬ: core_layer/predictor.py
#
# Назначение:
# - обёртка над выбранной моделью;
# - прямая задача: расчёт вероятности поражения;
# - пакетный прогноз для нескольких вариантов пуска.
# ============================================================

import numpy as np
import pandas as pd
from core_layer.preprocessing import FeaturePreprocessor

class KillProbabilityPredictor:
    """
    Прямая задача: P_kill = f(H, D, V, N, C, T, метеоусловия).
    """

    def __init__(self, model, preprocessor: FeaturePreprocessor, config: dict):
        self.model = model
        self.preprocessor = preprocessor
        self.config = config

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Принимает DataFrame с исходными признаками.
        Возвращает массив вероятностей поражения.
        """
        required = set(self.config["features"])

        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Отсутствуют признаки: {missing}")

        X = self.preprocessor.transform(df)
        preds = self.model.predict(X)

        return np.clip(preds, 0.0, 1.0)

    def predict_single(
        self,
        H: float,
        D: float,
        V: float,
        N: int,
        C: str,
        T: str,
        W_dir: float,
        W_spd: float,
        Temp: float
    ) -> float:
        """
        Прогноз для одного набора параметров.
        """
        row = pd.DataFrame([{
            "H": H,
            "D": D,
            "V": V,
            "N": N,
            "C": C,
            "T": T,
            "W_dir": W_dir,
            "W_spd": W_spd,
            "Temp": Temp
        }])

        return float(self.predict(row)[0])


print("Модуль core_layer.predictor определён.")