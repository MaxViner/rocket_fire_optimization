# ============================================================
# МОДУЛЬ: data_layer/data_validator.py
#
# Назначение:
# - проверка структуры данных;
# - удаление дубликатов;
# - обработка пропусков;
# - проверка диапазонов значений;
# - нормализация категориальных справочников.
# ============================================================


import pandas as pd
import numpy as np


class DataValidator:
    """
    Проверяет и очищает входной DataFrame.
    """

    def __init__(self, config: dict):
        self.config = config

        self.feature_columns = config["features"]
        self.target_column = config["target"]

        self.numeric_features = config["numeric_features"]
        self.categorical_features = config["categorical_features"]

    def validate(
        self,
        df: pd.DataFrame,
        mode: str = "training"
    ) -> pd.DataFrame:
        """
        Валидирует и очищает DataFrame.

        mode:
            "training"  — требуются признаки и целевая переменная P_kill;
            "inference" — требуются только признаки, P_kill не обязателен.
        """
        if mode not in ("training", "inference"):
            raise ValueError(
                "Некорректный режим валидации. "
                "Допустимые значения: 'training', 'inference'."
            )

        df = df.copy()

        required_columns = self.feature_columns.copy()

        if mode == "training":
            required_columns.append(self.target_column)

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Отсутствуют обязательные столбцы: {missing}"
            )

        # Удаляем полные дубликаты
        df = df.drop_duplicates()

        # Приводим числовые признаки к числовому типу
        for column in self.numeric_features:
            if column in df.columns:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

        # Нормализуем категориальные признаки
        for column in self.categorical_features:
            if column in df.columns:
                df[column] = (
                    df[column]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )

        # Удаляем строки с пропусками в обязательных полях
        df = df.dropna(subset=required_columns)

        # Ограничиваем значения допустимыми диапазонами
        df = self._clip_numeric_values(df)

        # Удаляем неизвестные категории
        df = self._filter_known_categories(df)

        return df.reset_index(drop=True)

    def _clip_numeric_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ограничивает числовые значения допустимыми диапазонами.
        """
        constraints = self.config["constraints"]

        for column, bounds in constraints.items():
            if column in df.columns:
                df[column] = df[column].clip(
                    lower=bounds["min"],
                    upper=bounds["max"]
                )

        return df

    def _filter_known_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Удаляет строки с неизвестными категориями.
        """
        known_targets = set(self.config["target_types"])
        known_effects = set(self.config["effect_types"])

        if "C" in df.columns:
            df = df[df["C"].isin(known_targets)]

        if "T" in df.columns:
            df = df[df["T"].isin(known_effects)]

        return df.reset_index(drop=True)

    def report(
        self,
        df: pd.DataFrame,
        mode: str = "training"
    ) -> dict:
        """
        Формирует краткий отчёт о качестве данных.
        """
        result = {
            "rows": len(df),
            "columns": list(df.columns),
            "missing_values": int(df.isna().sum().sum()),
            "duplicates": int(df.duplicated().sum())
        }

        if mode == "training" and self.target_column in df.columns:
            result["target_min"] = float(df[self.target_column].min())
            result["target_max"] = float(df[self.target_column].max())

        return result


print("Модуль data_layer.data_validator определён.")