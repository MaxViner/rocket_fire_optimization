# ============================================================
# ФАЙЛ: data_layer/data_loader.py
#
# Назначение:
# - загрузка данных из CSV, Excel, JSON;
# - сохранение данных в каталог storage;
# - генерация синтетического датасета;
# - генерация данных через SDV при наличии исходной выборки.
#
# Модуль не зависит от моделей и оптимизатора.
# ============================================================


from pathlib import Path

import numpy as np
import pandas as pd

from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

from config import CONFIG


class DataLoader:
    """
    Единая точка загрузки, генерации и сохранения табличных данных.
    """

    STORAGE_DIR = Path(__file__).resolve().parents[1] / "storage"

    # --------------------------------------------------------
    # Загрузка данных
    # --------------------------------------------------------
    @staticmethod
    def load(path: str) -> pd.DataFrame:
        """
        Загружает данные по расширению файла.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")

        suffix = path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(path)

        if suffix in (".xls", ".xlsx"):
            return pd.read_excel(path)

        if suffix == ".json":
            return pd.read_json(path)

        raise ValueError(f"Неподдерживаемый формат файла: {suffix}")

    # --------------------------------------------------------
    # Сохранение данных
    # --------------------------------------------------------
    @staticmethod
    def save(
        df: pd.DataFrame,
        filename: str,
        fmt: str = "csv"
    ) -> Path:
        """
        Сохраняет DataFrame в папку storage.

        fmt: "csv", "xlsx" или "json"
        """
        DataLoader.STORAGE_DIR.mkdir(parents=True, exist_ok=True)

        fmt = fmt.lower().lstrip(".")
        filepath = DataLoader.STORAGE_DIR / f"{filename}.{fmt}"

        if fmt == "csv":
            df.to_csv(
                filepath,
                index=False,
                encoding="utf-8-sig"
            )

        elif fmt in ("xls", "xlsx"):
            df.to_excel(filepath, index=False)

        elif fmt == "json":
            df.to_json(
                filepath,
                orient="records",
                force_ascii=False,
                indent=2
            )

        else:
            raise ValueError(f"Неподдерживаемый формат сохранения: {fmt}")

        return filepath

    # --------------------------------------------------------
    # Генерация синтетического датасета
    # --------------------------------------------------------
    @staticmethod
    def generate_synthetic(
        n_samples: int = 5000,
        seed: int = 42,
        save: bool = True,
        filename: str = "synthetic_data",
        fmt: str = "csv"
    ) -> pd.DataFrame:
        """
        Генерирует синтетический датасет с калиброванной вероятностью поражения.

        Целевая калибровка:
        при N = N_REF, H = OPTIMAL_HEIGHT, D = 2500 м,
        C = "stronghold", T = "destroy", W_spd = 10 м/с
        вероятность поражения составляет около 0.90.
        """
        rng = np.random.default_rng(seed)

        params = CONFIG["BASE_SYNTHETIC_PARAMS"]

        n_ref = params["N_REF"]
        kill_calibration = params["KILL_CALIBRATION"]

        dist_scale = params["DIST_SCALE"]
        height_scale = params["HEIGHT_SCALE"]
        optimal_height = params["OPTIMAL_HEIGHT"]
        max_wind_impact = params["MAX_WIND_IMPACT"]

        target_effect_matrix = params["TARGET_EFFECT_MATRIX"]

        # ----------------------------------------------------
        # Входные признаки
        # ----------------------------------------------------
        H = rng.uniform(50, 2000, n_samples)
        D = rng.uniform(1000, 6000, n_samples)
        V = rng.uniform(160, 280, n_samples)
        N = rng.integers(1, 20, n_samples)

        C = rng.choice(CONFIG["target_types"], n_samples)
        T = rng.choice(CONFIG["effect_types"], n_samples)

        W_dir = rng.uniform(0, 360, n_samples)
        W_spd = rng.uniform(0, 20, n_samples)
        Temp = rng.uniform(-30, 40, n_samples)

        # ----------------------------------------------------
        # Фактор количества ракет
        # ----------------------------------------------------
        # При N = N_REF фактор равен 1.
        base_effect = N / n_ref

        # ----------------------------------------------------
        # Фактор дальности
        # ----------------------------------------------------
        dist_factor = np.clip(
            1 - (D / dist_scale) ** 2,
            0.55,
            1.0
        )

        # ----------------------------------------------------
        # Фактор высоты
        # ----------------------------------------------------
        height_factor = np.clip(
            1 - (np.abs(H - optimal_height) / height_scale) ** 2,
            0.70,
            1.0
        )

        # ----------------------------------------------------
        target_effect_factor = np.array([
            target_effect_matrix[target_type][effect_type]
            for target_type, effect_type in zip(C, T)
        ])

        # ----------------------------------------------------
        # Фактор ветра
        # ----------------------------------------------------
        weather_factor = 1 - np.clip(
            W_spd / 20,
            0,
            1
        ) * max_wind_impact

        # ----------------------------------------------------
        # Нормированная эффективность залпа
        # ----------------------------------------------------
        effectiveness = (
            base_effect
            * dist_factor
            * height_factor
            * target_effect_factor
            * weather_factor
        )

        # ----------------------------------------------------
        # Вероятность поражения
        # ----------------------------------------------------
        # P_kill = 1 - exp(-KILL_CALIBRATION * effectiveness)
        #
        # Форма обеспечивает:
        # - P_kill > 0;
        # - P_kill < 1;
        # - рост вероятности с увеличением N;
        # - насыщение вероятности при большом N.
        # ----------------------------------------------------
        noise = rng.normal(0, 0.013, n_samples)

        P_kill = 1 - np.exp(
            -kill_calibration * effectiveness
        ) + noise

        P_kill = np.clip(P_kill, 0.01, 0.99)

        df = pd.DataFrame({
            "H": H,
            "D": D,
            "V": V,
            "N": N,
            "C": C,
            "T": T,
            "W_dir": W_dir,
            "W_spd": W_spd,
            "Temp": Temp,
            "P_kill": P_kill,
            "P_req": target_effect_factor
        })

        if save:
            filepath = DataLoader.save(df, filename, fmt)
            print(f"Синтетические данные сохранены: {filepath}")

        return df

    # --------------------------------------------------------
    # Генерация через SDV
    # --------------------------------------------------------
    @staticmethod
    def generate_with_sdv(
        real_data: pd.DataFrame,
        n_samples: int = 5000,
        seed: int = 42,
        save: bool = True,
        filename: str = "sdv_synthetic_data",
        fmt: str = "csv"
    ) -> pd.DataFrame:
        """
        Генерация синтетики через SDV.

        Требует установленного пакета sdv.
        """
        metadata = Metadata.detect_from_dataframe(
            data=real_data,
            table_name="dataset"
        )

        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(real_data)

        synthetic = synthesizer.sample(num_rows=n_samples)

        if save:
            filepath = DataLoader.save(synthetic, filename, fmt)
            print(f"SDV-синтетические данные сохранены: {filepath}")

        return synthetic


print("Модуль data_layer.data_loader определён.")