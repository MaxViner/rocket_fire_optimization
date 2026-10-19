# ============================================================
# ФАЙЛ: config.py
# Общая конфигурация проекта.
# ============================================================

from pathlib import Path

CONFIG = {
    "artifacts_dir": Path("artifacts"),
    "reports_dir": Path("artifacts/reports"),
    "models_dir": Path("artifacts/models"),

    "target": "P_kill",

    "numeric_features": ["H", "D", "V", "N", "W_dir", "W_spd", "Temp"],
    "categorical_features": ["C", "T"],
    "features": [
        "H", "D", "V", "N", "W_dir", "W_spd", "Temp", "C", "T"
    ],

    "test_size": 0.2,
    "random_state": 10,

    "cv_folds": 3,
    "optuna_trials": 20,

    "models": {
        "catboost": {
            "class": "CatBoostRegressor",
            "params": {
                "iterations": 300,
                "depth": 6,
                "learning_rate": 0.08,
                "loss_function": "RMSE",
                "verbose": False,
                "random_state": 42
            }
        },
        "random_forest": {
            "class": "RandomForestRegressor",
            "params": {
                "n_estimators": 200,
                "max_depth": 12,
                "random_state": 42,
                "n_jobs": -1
            }
        },
        "neural_network": {
            "class": "TorchRegressor",
            "params": {
                "hidden_dim": 64,
                "epochs": 80,
                "batch_size": 64,
                "learning_rate": 0.001,
                "dropout": 0.2
            }
        },
        "stacking": {
    "class": "StackingRegressor",
    "params": {
        "meta_model": "ridge"
    }
}
    },

    "constraints": {
        "H": {"min": 50, "max": 2000},
        "D": {"min": 500, "max": 6000},
        "V": {"min": 160, "max": 300},
        "N": {"min": 1, "max": 20}
    },

    "target_types": ["stronghold", "infantry", "apc", "command_post"],
    "effect_types": ["destroy", "damage", "suppress"],

   "BASE_SYNTHETIC_PARAMS": {
        "N_REF": 4,                   # эталонное количество ракет
        "KILL_CALIBRATION": 2.5,      # калибровочный коэффициент P_kill

        "DIST_SCALE": 8000,          # условная максимальная дальность, м
        "HEIGHT_SCALE": 4000,         # масштаб отклонения высоты, м
        "OPTIMAL_HEIGHT": 500,       # оптимальная высота пуска, м
        "MAX_WIND_IMPACT": 0.05,      # максимальное относительное влияние ветра

        "TARGET_EFFECT_MATRIX": {
            "infantry": {
                "suppress": 1.00,
                "damage": 0.90,
                "destroy": 0.75
            },
            "apc": {
                "suppress": 0.95,
                "damage": 0.75,
                "destroy": 0.55
            },
            "command_post": {
                "suppress": 0.90,
                "damage": 0.70,
                "destroy": 0.50
            },
            "stronghold": {
                "suppress": 0.80,
                "damage": 0.60,
                "destroy": 0.40
            }
}
    },
    "best_model_name": "best_model",
}


def create_directories():
    CONFIG["artifacts_dir"].mkdir(exist_ok=True)
    CONFIG["reports_dir"].mkdir(exist_ok=True)
    CONFIG["models_dir"].mkdir(exist_ok=True)