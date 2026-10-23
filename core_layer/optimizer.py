# ============================================================
# МОДУЛЬ: core_layer/optimizer.py
#
# Быстрый оптимизатор обратной задачи.
#
# Стратегия:
# 1. Грубый поиск по сетке;
# 2. Пакетный прогноз всех комбинаций;
# 3. Ранняя остановка по минимальному N;
# 4. Локальное уточнение вокруг лучших грубых решений.
# ============================================================

import itertools
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from core_layer.constraint_checker import ConstraintChecker
from core_layer.killProbabilityPredictor import KillProbabilityPredictor


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
        search_ranges: dict = None,
        coarse_factor: int = 4,
        show_progress: bool = True
    ):
        """
        Выполняет двухэтапную оптимизацию.

        coarse_factor -- во сколько раз укрупняется грубая сетка.
        Например, при step=250 и coarse_factor=4 шаг грубого поиска
        составит 1000.
        """
        search_ranges = search_ranges or {
            "H": {"min": 50, "max": 2000, "step": 250},
            "D": {"min": 500, "max": 6000, "step": 500},
            "V": {"min": 160, "max": 300, "step": 25},
            "N": {"min": 1, "max": 20, "step": 1}
        }

        start_time = time.perf_counter()

        # ----------------------------------------------------
        # ЭТАП 1. Грубый поиск
        # ----------------------------------------------------
        coarse_ranges = self._make_coarse_ranges(
            search_ranges,
            coarse_factor
        )

        coarse_result = self._grid_search(
            target_type=target_type,
            effect_type=effect_type,
            weather=weather,
            p_required=p_required,
            search_ranges=coarse_ranges,
            stage_name="Грубый поиск",
            show_progress=show_progress
        )

        if not coarse_result["found"]:
            return {
                "found": False,
                "solutions": pd.DataFrame(),
                "best": None,
                "message": (
                    "Допустимое решение не найдено "
                    "в пределах заданных диапазонов поиска."
                ),
                "elapsed_sec": time.perf_counter() - start_time,
                "checked_combinations": coarse_result["checked_combinations"]
            }

        # ----------------------------------------------------
        # ЭТАП 2. Локальное уточнение
        # ----------------------------------------------------
        fine_ranges = self._make_local_ranges(
            search_ranges=search_ranges,
            center=coarse_result["best"]
        )

        fine_result = self._grid_search(
            target_type=target_type,
            effect_type=effect_type,
            weather=weather,
            p_required=p_required,
            search_ranges=fine_ranges,
            stage_name="Локальное уточнение",
            show_progress=show_progress
        )

        elapsed = time.perf_counter() - start_time

        final_result = fine_result if fine_result["found"] else coarse_result

        return {
            "found": final_result["found"],
            "solutions": final_result["solutions"],
            "best": final_result["best"],
            "elapsed_sec": elapsed,
            "checked_combinations": (
                coarse_result["checked_combinations"]
                + fine_result["checked_combinations"]
            ),
            "coarse_best": coarse_result["best"],
            "search_strategy": "coarse_to_fine"
        }

    # --------------------------------------------------------
    # Внутренние методы
    # --------------------------------------------------------
    def _grid_search(
        self,
        target_type,
        effect_type,
        weather,
        p_required,
        search_ranges,
        stage_name,
        show_progress
    ):
        """
        Пакетный поиск по сетке для минимального N.
        """
        H_values = self._make_grid(
            search_ranges["H"]["min"],
            search_ranges["H"]["max"],
            search_ranges["H"]["step"]
        )

        D_values = self._make_grid(
            search_ranges["D"]["min"],
            search_ranges["D"]["max"],
            search_ranges["D"]["step"]
        )

        V_values = self._make_grid(
            search_ranges["V"]["min"],
            search_ranges["V"]["max"],
            search_ranges["V"]["step"]
        )

        N_values = range(
            int(search_ranges["N"]["min"]),
            int(search_ranges["N"]["max"]) + 1,
            int(search_ranges["N"].get("step", 1))
        )

        combinations = np.array(
            list(itertools.product(H_values, D_values, V_values)),
            dtype=float
        )

        total_combinations = len(combinations)
        total_checks = total_combinations * len(N_values)

        progress_bar = None

        if show_progress:
            progress_bar = tqdm(
                total=total_checks,
                desc=stage_name,
                unit="вар.",
                ncols=110
            )

        feasible = []

        try:
            for N in N_values:

                scenarios = pd.DataFrame({
                    "H": combinations[:, 0],
                    "D": combinations[:, 1],
                    "V": combinations[:, 2],
                    "N": np.full(total_combinations, N, dtype=int),
                    "C": target_type,
                    "T": effect_type,
                    "W_dir": weather["dir"],
                    "W_spd": weather["spd"],
                    "Temp": weather["temp"]
                })

                # Пакетный прогноз вместо цикла по отдельным точкам
                probabilities = self.predictor.predict_batch(scenarios)

                mask = (
                    (probabilities >= p_required)
                    & self._get_valid_mask(
                        H=combinations[:, 0],
                        D=combinations[:, 1],
                        V=combinations[:, 2],
                        N=N,
                        target_type=target_type,
                        effect_type=effect_type
                    )
                )

                valid_indices = np.where(mask)[0]

                for i in valid_indices:
                    feasible.append({
                        "N": N,
                        "H": combinations[i, 0],
                        "D": combinations[i, 1],
                        "V": combinations[i, 2],
                        "P_kill": probabilities[i],
                        "P_required": p_required,
                        "is_valid": True,
                        "violations": ""
                    })

                if progress_bar is not None:
                    progress_bar.update(total_combinations)
                    progress_bar.set_postfix({
                        "N": N,
                        "найдено": len(feasible)
                    })

                # Минимальное N уже найдено — прекращаем перебор
                if feasible:
                    break

        finally:
            if progress_bar is not None:
                progress_bar.close()

        if not feasible:
            return {
                "found": False,
                "solutions": pd.DataFrame(),
                "best": None,
                "checked_combinations": total_checks
            }

        solutions = pd.DataFrame(feasible)

        min_n = solutions["N"].min()
        solutions = solutions[solutions["N"] == min_n]

        best = solutions.sort_values(
            by=["P_kill", "D"],
            ascending=[False, True]
        ).iloc[0].to_dict()

        return {
            "found": True,
            "solutions": solutions.reset_index(drop=True),
            "best": best,
            "checked_combinations": total_checks
        }

    def _get_valid_mask(
        self,
        H,
        D,
        V,
        N,
        target_type,
        effect_type
    ):
        """
        Векторная проверка ограничений.
        """
        constraints = self.config["constraints"]

        valid = (
            (H >= constraints["H"]["min"]) & (H <= constraints["H"]["max"])
            & (D >= constraints["D"]["min"]) & (D <= constraints["D"]["max"])
            & (V >= constraints["V"]["min"]) & (V <= constraints["V"]["max"])
            & (N >= constraints["N"]["min"]) & (N <= constraints["N"]["max"])
            & (target_type in self.config["target_types"])
            & (effect_type in self.config["effect_types"])
        )

        return valid

    def _make_coarse_ranges(
        self,
        search_ranges,
        coarse_factor: int
    ):
        """
        Укрупняет шаг сетки для первого этапа поиска.
        """
        coarse = {}

        for key, value in search_ranges.items():
            coarse[key] = {
                "min": value["min"],
                "max": value["max"],
                "step": max(
                    value["step"] * coarse_factor,
                    (value["max"] - value["min"]) / 20
                )
            }

        return coarse

    def _make_local_ranges(
        self,
        search_ranges,
        center: dict
    ):
        """
        Формирует локальную область поиска вокруг лучшего грубого решения.
        """
        local = {}

        for key in ["H", "D", "V"]:
            step = search_ranges[key]["step"]
            value = center[key]

            local[key] = {
                "min": max(
                    search_ranges[key]["min"],
                    value - step
                ),
                "max": min(
                    search_ranges[key]["max"],
                    value + step
                ),
                "step": step
            }

        local["N"] = {
            "min": int(center["N"]),
            "max": int(center["N"]),
            "step": 1
        }

        return local

    @staticmethod
    def _make_grid(min_value, max_value, step):
        """
        Формирует сетку значений от min до max.
        """
        return np.arange(
            min_value,
            max_value + step,
            step
        ).tolist()