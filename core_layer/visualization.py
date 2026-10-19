# ============================================================
# МОДУЛЬ: core_layer/visualization.py
#
# Назначение:
# - построение русскоязычных графиков для отчёта и ВКР;
# - сравнение моделей;
# - train/test-диагностика;
# - кривые обучения;
# - график зависимости вероятности поражения от числа ракет;
# - сохранение графиков в storage/figures.
# ============================================================


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------
# Единые настройки оформления графиков
# ------------------------------------------------------------
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300

MODEL_NAMES_RU = {
    "random_forest": "Случайный лес",
    "catboost": "CatBoost",
    "neural_network": "Нейронная сеть",
    "stacking": "Стекинг"
}

METRIC_NAMES_RU = {
    "R2": "Коэффициент детерминации R²",
    "MAE": "Средняя абсолютная ошибка",
    "MSE": "Среднеквадратичная ошибка",
    "RMSE": "Среднеквадратичное отклонение",
    "inference_time_sec": "Время инференса, с"
}


class Visualizer:
    """
    Класс для построения и сохранения аналитических графиков.
    """

    FIGURES_DIR = (
        Path(__file__).resolve().parents[1]
        / "storage"
        / "figures"
    )

    def __init__(self, style: str = "whitegrid"):
        sns.set_style(style)
        self.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Вспомогательные методы
    # --------------------------------------------------------
    def _translate_model_name(self, model_name: str) -> str:
        """
        Переводит техническое имя модели на русский язык.
        """
        return MODEL_NAMES_RU.get(model_name, model_name)

    def _translate_metric_name(self, metric: str) -> str:
        """
        Переводит техническое название метрики на русский язык.
        """
        return METRIC_NAMES_RU.get(metric, metric)

    def _prepare_comparison_dataframe(
        self,
        comparison_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Добавляет русскоязычное название модели.
        """
        df = comparison_df.copy()
        df["Модель"] = df["model"].apply(self._translate_model_name)
        return df

    def _finish_figure(
        self,
        fig,
        filename: str,
        show: bool = False,
        dpi: int = 300
    ) -> Path:
        """
        Сохраняет график и закрывает фигуру.
        """
        filepath = self.FIGURES_DIR / filename

        fig.savefig(
            filepath,
            dpi=dpi,
            bbox_inches="tight"
        )

        print(f"График сохранён: {filepath}")

        if show:
            plt.show(block=False)
            plt.pause(0.1)

        plt.close(fig)

        return filepath

    # --------------------------------------------------------
    # Сравнение моделей
    # --------------------------------------------------------
    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric: str = "R2",
        palette: str = "viridis",
        ylim: tuple = None,
        show: bool = False,
        filename: str = None
    ):
        """
        Строит столбчатую диаграмму сравнения моделей.
        """
        if comparison_df is None or comparison_df.empty:
            raise ValueError("Нет данных для визуализации.")

        if metric not in comparison_df.columns:
            raise ValueError(
                f"Метрика '{metric}' отсутствует. "
                f"Доступны: {list(comparison_df.columns)}"
            )

        df = self._prepare_comparison_dataframe(comparison_df)
        metric_name_ru = self._translate_metric_name(metric)

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(
            data=df,
            x="Модель",
            y=metric,
            hue="Модель",
            palette=palette,
            legend=False,
            ax=ax
        )

        ax.set_title(
            f"Сравнение моделей по метрике «{metric_name_ru}»"
        )
        ax.set_xlabel("Модель")
        ax.set_ylabel(metric_name_ru)
        ax.grid(axis="y", alpha=0.3)

        if ylim is not None:
            ax.set_ylim(*ylim)

        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.3f",
                padding=3,
                fontsize=9
            )

        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        fig.tight_layout()

        if filename is None:
            filename = f"model_comparison_{metric}.png"

        return self._finish_figure(
            fig=fig,
            filename=filename,
            show=show
        )

    # --------------------------------------------------------
    # Train / Test-диагностика
    # --------------------------------------------------------
    def plot_train_test_comparison(
        self,
        train_test_df: pd.DataFrame,
        show: bool = False,
        filename: str = "sravnenie_kachestva_train_test.png"
    ):
        """
        Строит сравнение R² на обучающей и тестовой выборках.
        """
        if train_test_df is None or train_test_df.empty:
            raise ValueError("Нет данных для визуализации.")

        df = self._prepare_comparison_dataframe(train_test_df)

        fig, ax = plt.subplots(figsize=(11, 6))

        x = np.arange(len(df))
        width = 0.35

        bars_train = ax.bar(
            x - width / 2,
            df["train_R2"],
            width,
            label="Обучающая выборка",
            color="#4C72B0"
        )

        bars_test = ax.bar(
            x + width / 2,
            df["test_R2"],
            width,
            label="Тестовая выборка",
            color="#DD8452"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(df["Модель"], rotation=15, ha="right")
        ax.set_ylabel("Коэффициент детерминации R²")
        ax.set_title(
            "Сравнение качества моделей на обучающей и тестовой выборках"
        )
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

        ax.bar_label(bars_train, fmt="%.3f", padding=3, fontsize=9)
        ax.bar_label(bars_test, fmt="%.3f", padding=3, fontsize=9)

        fig.tight_layout()

        return self._finish_figure(
            fig=fig,
            filename=filename,
            show=show
        )

    def plot_predictions_vs_actual(
        self,
        y_true,
        y_pred,
        title: str = "Сопоставление фактических и прогнозных значений",
        show: bool = False,
        filename: str = "fakt_i_prognoz.png"
    ):
        """
        Строит диаграмму «факт — прогноз».
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        fig, ax = plt.subplots(figsize=(7, 7))

        ax.scatter(
            y_true,
            y_pred,
            alpha=0.45,
            s=22,
            color="#4C72B0",
            label="Наблюдения"
        )

        ax.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Идеальное совпадение"
        )

        ax.set_xlabel("Фактическая вероятность поражения")
        ax.set_ylabel("Прогнозная вероятность поражения")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()

        fig.tight_layout()

        return self._finish_figure(
            fig=fig,
            filename=filename,
            show=show
        )

    def plot_residuals(
        self,
        y_true,
        y_pred,
        title: str = "Анализ остатков модели",
        show: bool = False,
        filename: str = "ostatki.png"
    ):
        """
        Строит график остатков.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        residuals = y_true - y_pred

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.scatter(
            y_pred,
            residuals,
            alpha=0.45,
            s=22,
            color="#4C72B0",
            label="Наблюдения"
        )

        ax.axhline(
            0,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Нулевая ошибка"
        )

        ax.set_xlabel("Прогнозная вероятность поражения")
        ax.set_ylabel("Остаток: факт − прогноз")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()

        fig.tight_layout()

        return self._finish_figure(
            fig=fig,
            filename=filename,
            show=show
        )

    # --------------------------------------------------------
    # Кривые обучения
    # --------------------------------------------------------
    def plot_learning_curves(
        self,
        learning_curves: dict,
        metric_name: str = "R2",
        show: bool = False,
        filename: str = "krivye_obucheniya.png"
    ):
        """
        Строит кривые обучения для всех моделей.
        """
        if not learning_curves:
            raise ValueError("Нет данных для построения кривых обучения.")

        metric_name_ru = self._translate_metric_name(metric_name)

        fig, ax = plt.subplots(figsize=(11, 6))

        for model_name, curve in learning_curves.items():
            model_name_ru = self._translate_model_name(model_name)

            train_sizes = curve["train_sizes"]
            train_scores = curve["train_scores_mean"]
            val_scores = curve["val_scores_mean"]

            ax.plot(
                train_sizes,
                train_scores,
                marker="o",
                linestyle="--",
                linewidth=1.8,
                label=f"{model_name_ru} — обучение"
            )

            ax.plot(
                train_sizes,
                val_scores,
                marker="s",
                linewidth=1.8,
                label=f"{model_name_ru} — валидация"
            )

        ax.set_xlabel("Количество обучающих примеров")
        ax.set_ylabel(metric_name_ru)
        ax.set_title(
            f"Кривые обучения моделей по метрике «{metric_name_ru}»"
        )
        ax.grid(alpha=0.3)
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=True
        )

        fig.tight_layout()

        return self._finish_figure(
            fig=fig,
            filename=filename,
            show=show
        )

    def plot_model_learning_curve(
        self,
        model_name: str,
        train_sizes,
        train_scores_mean,
        val_scores_mean,
        metric_name: str = "R2",
        show: bool = False,
        filename: str = None
    ):
        """
        Строит кривую обучения одной модели.
        """
        model_name_ru = self._translate_model_name(model_name)
        metric_name_ru = self._translate_metric_name(metric_name)

        if filename is None:
            filename = f"krivaya_obucheniya_{model_name}.png"

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(
            train_sizes,
            train_scores_mean,
            marker="o",
            linestyle="--",
            linewidth=1.8,
            color="#4C72B0",
            label="Обучающая выборка"
        )

        ax.plot(
            train_sizes,
            val_scores_mean,
            marker="s",
            linewidth=1.8,
            color="#DD8452",
            label="Валидационная выборка"
        )

        ax.fill_between(
            train_sizes,
            val_scores_mean,
            train_scores_mean,
            alpha=0.15,
            color="#4C72B0",
            label="Разрыв между обучением и валидацией"
        )

        ax.set_xlabel("Количество обучающих примеров")
        ax.set_ylabel(metric_name_ru)
        ax.set_title(
            f"Кривая обучения модели «{model_name_ru}»"
        )
        ax.grid(alpha=0.3)
        ax.legend()

        fig.tight_layout()

        return self._finish_figure(
            fig=fig,
            filename=filename,
            show=show
        )

    # --------------------------------------------------------
    # Зависимость вероятности поражения от числа ракет
    # --------------------------------------------------------
    def plot_probability_vs_rockets(
        self,
        N_values,
        probabilities,
        p_required: float,
        optimal_n: int,
        title: str = (
            "Зависимость вероятности поражения "
            "от количества ракет"
        ),
        show: bool = False,
        filename: str = "zavisimost_P_kill_ot_N.png"
    ):
        """
        Строит график P_kill(N).
        """
        if len(N_values) != len(probabilities):
            raise ValueError(
                "Длины N_values и probabilities должны совпадать."
            )

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(
            N_values,
            probabilities,
            marker="o",
            linewidth=2.2,
            color="#4C72B0",
            label="Расчётная вероятность поражения"
        )

        ax.axhline(
            y=p_required,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Требуемая вероятность: {p_required:.2f}"
        )

        ax.axvline(
            x=optimal_n,
            color="green",
            linestyle=":",
            linewidth=1.8,
            label=f"Минимальное количество ракет: {int(optimal_n)}"
        )

        ax.set_xlabel("Количество ракет в залпе, шт.")
        ax.set_ylabel("Вероятность поражения цели")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()

        fig.tight_layout()

        return self._finish_figure(
            fig=fig,
            filename=filename,
            show=show
        )


print("Модуль core_layer.visualization определён.")