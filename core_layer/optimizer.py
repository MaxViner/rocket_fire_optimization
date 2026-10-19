# ============================================================
# МОДУЛЬ: core_layer/optimizer.py
#
# Назначение:
# - обратная задача: найти H, D, V и минимальное N;
# - обеспечить P_kill >= P_required;
# - учитывать ограничения через ConstraintChecker;
# - использовать Predictor для оценки каждого варианта.
#
# Поиск выполняется по сетке в пределах диапазонов,
# заданных пользователем.
# ============================================================


import itertools

import numpy as np
import pandas as pd

from core_layer.constraint_checker import ConstraintChecker
from core_layer.kill_Probability_Predictor import KillProbabilityPredictor


class MissionOptimizer:
    """
    Решает обратную задачу выбора параметров пуска.
    """

    def __init__(
        self,
        predictor: KillProbabilityPredictor,
        constraint_checker: ConstraintChecker,
        config: dict
    ):
        self.predictor = predictor
        self.checker = constraint_checker
        self.config = config

    def optimize(
        self,
        target_type: str,
        effect_type: str,
        weather: dict,
        p_required: float,
        grid_step: dict = None,
        fixed_values: dict = None,
        search_ranges: dict = None
    ):
        """
        Ищет минимальное N и допустимые H, D, V.

        search_ranges задаёт диапазон условий боевого применения:
        {
            "H": (100, 2000),
            "D": (1000, 6000),
            "V": (160, 280)
        }

        grid_step задаёт шаг перебора:
        {
            "H": 250,
            "D": 500,
            "V": 20
        }

        fixed_values позволяет зафиксировать часть параметров:
        {
            "V": 250
        }
        """
        grid_step = grid_step or {
            "H": 250,
            "D": 500,
            "V": 20
        }

        fixed_values = fixed_values or {}
        search_ranges = search_ranges or {}

        constraints = self.config["constraints"]

        # ----------------------------------------------------
        # Определение диапазонов поиска
        # ----------------------------------------------------
        H_min, H_max = search_ranges.get(
            "H",
            (
                constraints["H"]["min"],
                constraints["H"]["max"]
            )
        )

        D_min, D_max = search_ranges.get(
            "D",
            (
                constraints["D"]["min"],
                constraints["D"]["max"]
            )
        )

        V_min, V_max = search_ranges.get(
            "V",
            (
                constraints["V"]["min"],
                constraints["V"]["max"]
            )
        )

        # ----------------------------------------------------
        # Проверка корректности диапазонов
        # ----------------------------------------------------
        if H_min > H_max:
            raise ValueError("Некорректный диапазон H: min больше max.")

        if D_min > D_max:
            raise ValueError("Некорректный диапазон D: min больше max.")

        if V_min > V_max:
            raise ValueError("Некорректный диапазон V: min больше max.")

        # ----------------------------------------------------
        # Формирование сеток
        # ----------------------------------------------------
        H_values = self._make_grid(
            H_min,
            H_max,
            grid_step["H"]
        )

        D_values = self._make_grid(
            D_min,
            D_max,
            grid_step["D"]
        )

        V_values = self._make_grid(
            V_min,
            V_max,
            grid_step["V"]
        )

        # Фиксированные значения имеют приоритет над диапазоном
        if "H" in fixed_values:
            H_values = [fixed_values["H"]]

        if "D" in fixed_values:
            D_values = [fixed_values["D"]]

        if "V" in fixed_values:
            V_values = [fixed_values["V"]]

        feasible = []

        # ----------------------------------------------------
        # Перебор количества ракет
        # ----------------------------------------------------
        # N перебирается от минимального к максимальному.
        # Как только для текущего N найдено хотя бы одно решение,
        # дальнейшее увеличение N не требуется: ищем минимальный расход.
        # ----------------------------------------------------
        for n in range(
            constraints["N"]["min"],
            constraints["N"]["max"] + 1
        ):
            for h, d, v in itertools.product(
                H_values,
                D_values,
                V_values
            ):
                check = self.checker.check_single(
                    H=h,
                    D=d,
                    V=v,
                    N=n,
                    C=target_type,
                    T=effect_type
                )

                if not check["is_valid"]:
                    continue

                p_kill = self.predictor.predict_single(
                    H=h,
                    D=d,
                    V=v,
                    N=n,
                    C=target_type,
                    T=effect_type,
                    W_dir=weather["dir"],
                    W_spd=weather["spd"],
                    Temp=weather["temp"]
                )

                if p_kill >= p_required:
                    feasible.append({
                        "N": n,
                        "H": h,
                        "D": d,
                        "V": v,
                        "P_kill": p_kill,
                        "P_required": p_required,
                        "C": target_type,
                        "T": effect_type,
                        "W_dir": weather["dir"],
                        "W_spd": weather["spd"],
                        "Temp": weather["temp"],
                        "is_valid": True,
                        "violations": ""
                    })

            # Все допустимые варианты для минимального N найдены
            if feasible:
                break

        if not feasible:
            return {
                "found": False,
                "solutions": pd.DataFrame(),
                "best": None,
                "message": (
                    "Допустимое решение не найдено "
                    "в заданном диапазоне условий."
                )
            }

        solutions = pd.DataFrame(feasible)

        # ----------------------------------------------------
        # Выбор лучшего решения
        # ----------------------------------------------------
        # Приоритеты:
        # 1. Минимальное N уже обеспечено остановкой перебора.
        # 2. Максимальная P_kill.
        # 3. Минимальная дальность D.
        # 4. Минимальная высота H.
        # ----------------------------------------------------
        best = solutions.sort_values(
            by=["P_kill", "D", "H"],
            ascending=[False, True, True]
        ).iloc[0].to_dict()

        return {
            "found": True,
            "solutions": solutions.reset_index(drop=True),
            "best": best,
            "message": "Оптимальное решение найдено."
        }

    @staticmethod
    def _make_grid(
        min_value: float,
        max_value: float,
        step: float
    ) -> list[float]:
        """
        Формирует сетку значений от min до max с заданным шагом.
        """
        if step <= 0:
            raise ValueError("Шаг сетки должен быть положительным.")

        values = np.arange(
            min_value,
            max_value + step,
            step
        )

        # Исключаем возможное превышение max из-за погрешности float
        values = values[values <= max_value + 1e-9]

        return values.tolist()


print("Модуль core_layer.optimizer определён.")