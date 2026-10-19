# Назначение:
# - кодирование категориальных признаков;
# - масштабирование числовых признаков;
# - подготовка train/test выборок;
# - опциональная балансировка SMOTE.
#
# Важно: SMOTE для регрессии применяется только как
# исследовательская опция, а не как обязательный этап.
# ============================================================

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split


class FeaturePreprocessor:
    """
    Преобразует исходные признаки в матрицу для моделей.
    """

    def __init__(self, config: dict):
        self.config = config
        self.numeric_features = config["numeric_features"]
        self.categorical_features = config["categorical_features"]

        self.scaler = StandardScaler()
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        self.fitted = False

    def fit_transform(self, df: pd.DataFrame):
        """
        Обучает преобразователи и возвращает обработанные данные.
        """
        df = df.copy()

        numeric = df[self.numeric_features].values.astype(float)
        categorical = df[self.categorical_features].values.astype(str)

        numeric_scaled = self.scaler.fit_transform(numeric)
        categorical_encoded = self.encoder.fit_transform(categorical)

        X = np.hstack([numeric_scaled, categorical_encoded])
        y = df[self.config["target"]].values.astype(float)

        self.fitted = True
        return X, y

    def transform(self, df: pd.DataFrame):
        """
        Преобразует новые данные без повторного обучения.
        """
        if not self.fitted:
            raise RuntimeError("Преобразователь не обучен. Сначала вызовите fit_transform().")

        df = df.copy()

        numeric = df[self.numeric_features].values.astype(float)
        categorical = df[self.categorical_features].values.astype(str)

        numeric_scaled = self.scaler.transform(numeric)
        categorical_encoded = self.encoder.transform(categorical)

        X = np.hstack([numeric_scaled, categorical_encoded])
        return X

    def split(self, X, y):
        """
        Разделяет данные на обучающую и тестовую выборки.
        """
        return train_test_split(
            X,
            y,
            test_size=self.config["test_size"],
            random_state=self.config["random_state"]
        )

    def apply_smote(self, X, y, threshold: float = 0.5):
        """
        Опциональная балансировка для регрессии через бинаризацию цели.
        Не рекомендуется использовать по умолчанию.
        """
        from imblearn.over_sampling import SMOTE

        y_binary = (y >= threshold).astype(int)

        if len(np.unique(y_binary)) < 2:
            return X, y

        smote = SMOTE(random_state=self.config["random_state"])
        X_res, y_binary_res = smote.fit_resample(X, y_binary)

        # Для регрессии возвращаем исходные y, соответствующие
        # синтетически увеличенной выборке невозможно корректно
        # восстановить без отдельной регрессионной модели.
        # Поэтому SMOTE здесь используется только для классификационных
        # экспериментов.
        return X_res, y_binary_res


print("Модуль core_layer.preprocessing определён.")