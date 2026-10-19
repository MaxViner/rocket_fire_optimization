# ЯЧЕЙКА 8. МОДУЛЬ: core_layer/constraint_checker.py
#
# Назначение:
# - проверка допустимости H, D, V, N;
# - проверка категориальных параметров;
# - формирование отчёта о нарушенных ограничениях.
# ============================================================

import pandas as pd


class ConstraintChecker:
    """
    Проверяет соответствие параметров пуска заданным ограничениям.
    """

    def __init__(self, config: dict):
        self.config = config
        self.constraints = config["constraints"]

    def check_single(
        self,
        H: float,
        D: float,
        V: float,
        N: int,
        C: str,
        T: str
    ) -> dict:
        """
        Проверяет один вариант решения.
        """
        violations = []

        checks = {
            "H": H,
            "D": D,
            "V": V,
            "N": N
        }

        for param, value in checks.items():
            bounds = self.constraints[param]

            if value < bounds["min"]:
                violations.append(
                    f"{param}={value} меньше минимума {bounds['min']}"
                )

            if value > bounds["max"]:
                violations.append(
                    f"{param}={value} больше максимума {bounds['max']}"
                )

        if C not in self.config["target_types"]:
            violations.append(f"Неизвестный тип цели: {C}")

        if T not in self.config["effect_types"]:
            violations.append(f"Неизвестный тип воздействия: {T}")

        return {
            "is_valid": len(violations) == 0,
            "violations": violations
        }

    def check_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Проверяет таблицу с вариантами решений.
        """
        results = []

        for _, row in df.iterrows():
            result = self.check_single(
                H=row["H"],
                D=row["D"],
                V=row["V"],
                N=int(row["N"]),
                C=row["C"],
                T=row["T"]
            )

            results.append(result)

        df = df.copy()
        df["is_valid"] = [x["is_valid"] for x in results]
        df["violations"] = [", ".join(x["violations"]) for x in results]

        return df


print("Модуль core_layer.constraint_checker определён.")