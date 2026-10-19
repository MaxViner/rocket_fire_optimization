# Назначение:
# - унифицированное обучение CatBoost, RandomForest и PyTorch NN;
# - кросс-валидация;
# - подбор гиперпараметров Optuna;
# - сохранение обученных моделей.
#
# Новый алгоритм добавляется через фабрику create_model().
# ============================================================

import time

import joblib
import numpy as np
import optuna
import torch
import torch.nn as nn
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin

class TorchRegressor(BaseEstimator, RegressorMixin):
    """
    Обёртка над нейросетью PyTorch с интерфейсом,
    совместимым с scikit-learn: fit() и predict().
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        epochs: int = 80,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        dropout: float = 0.2,
        random_state: int = 42
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.random_state = random_state

        self.params = {
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
            "random_state": self.random_state
        }

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.model = self._build_model()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def fit(self, X, y):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)

        dataset = TensorDataset(X_tensor, y_tensor)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        self.model.train()

        for _ in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            preds = self.model(X_tensor).cpu().numpy().flatten()

        return preds

    def save(self, path):
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_dim": self.input_dim,
                "params": self.params
            },
            path
        )


class ModelTrainer:
    """
    Унифицированный интерфейс обучения моделей.
    """

    def __init__(self, config: dict):
        self.config = config

    def create_model(self, model_name: str, input_dim: int):
        """
        Фабрика моделей.
        """
        if model_name not in self.config["models"]:
            raise ValueError(f"Неизвестная модель: {model_name}")

        model_config = self.config["models"][model_name]
        params = model_config["params"].copy()

        if model_name == "catboost":
            return CatBoostRegressor(**params)

        if model_name == "random_forest":
            return RandomForestRegressor(**params)

        if model_name == "neural_network":
            return TorchRegressor(
                input_dim=input_dim,
                **params
            )
        if model_name == "stacking":
            from sklearn.ensemble import StackingRegressor
            from sklearn.linear_model import RidgeCV

            base_models = [
                (
                    "catboost",
                    CatBoostRegressor(
                        iterations=300,
                        depth=6,
                        learning_rate=0.08,
                        loss_function="RMSE",
                        verbose=False,
                        random_state=42
                    )
                ),
                (
                    "random_forest",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=12,
                        random_state=42,
                        n_jobs=-1
                    )
                )
            ]

            return StackingRegressor(
                estimators=base_models,
                final_estimator=RidgeCV(),
                cv=5,
                n_jobs=-1
            )

        raise ValueError(f"Модель {model_name} не поддерживается.")

    def train(self, model_name: str, X_train, y_train):
        """
        Обучает модель и возвращает кортеж (model, training_time).
        """
        model = self.create_model(model_name, X_train.shape[1])

        start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        return model, elapsed

    def cross_validate(self, model_name: str, X, y):
        """
        Кросс-валидация для sklearn-совместимых моделей.
        Для нейросети выполняется упрощённая оценка.
        """
        if model_name == "neural_network":
            return None

        model = self.create_model(model_name, X.shape[1])

        scores = cross_val_score(
            model,
            X,
            y,
            cv=self.config["cv_folds"],
            scoring="r2",
            n_jobs=-1
        )

        return {
            "mean_r2": float(scores.mean()),
            "std_r2": float(scores.std()),
            "scores": scores.tolist()
        }

    def tune_hyperparameters(
        self,
        model_name: str,
        X_train,
        y_train,
        n_trials: int = None
    ):
        """
        Подбор гиперпараметров через Optuna.
        """
        n_trials = n_trials or self.config["optuna_trials"]

        def objective(trial):
            if model_name == "catboost":
                params = {
                    "iterations": trial.suggest_int("iterations", 100, 500),
                    "depth": trial.suggest_int("depth", 4, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "loss_function": "RMSE",
                    "verbose": False,
                    "random_state": self.config["random_state"]
                }
                model = CatBoostRegressor(**params)

            elif model_name == "random_forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                    "max_depth": trial.suggest_int("max_depth", 5, 20),
                    "random_state": self.config["random_state"],
                    "n_jobs": -1
                }
                model = RandomForestRegressor(**params)

            else:
                raise ValueError("Optuna-подбор для нейросети в базовой версии отключён.")

            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=3,
                scoring="r2",
                n_jobs=-1
            )

            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        return study.best_params, study.best_value

    def save_model(self, model, model_name: str):
        """
        Сохраняет модель в стандартизированном формате.
        """
        if isinstance(model, TorchRegressor):
            path = self.config["models_dir"] / f"{model_name}.pt"
            model.save(path)
        else:
            path = self.config["models_dir"] / f"{model_name}.joblib"
            joblib.dump(model, path)

        return path


print("Модуль core_layer.model_trainer определён.")