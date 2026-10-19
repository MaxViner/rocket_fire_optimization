# ============================================================
# МОДУЛЬ: application_layer/services/report_service.py
#
# Назначение:
# - формирование отчёта по результатам одного запуска;
# - сохранение таблиц, графиков и Markdown-отчёта;
# - обеспечение воспроизводимости результатов.
# ============================================================


from datetime import datetime
from pathlib import Path

import pandas as pd


class ReportService:
    """
    Формирует отчёт по результатам одного выполнения программы.
    """

    def __init__(self, config: dict):
        self.config = config

        self.reports_dir = Path(config["reports_dir"])
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.run_dir = None
        self.figures_dir = None
        self.report_sections = []

    def start_run(self) -> Path:
        """
        Создаёт отдельный каталог для текущего запуска.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = self.reports_dir / f"report_{timestamp}"
        self.figures_dir = self.run_dir / "figures"

        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.report_sections = []

        return self.run_dir

    def add_section(self, title: str, content: str):
        """
        Добавляет раздел в Markdown-отчёт.
        """
        self.report_sections.append(
            f"## {title}\n\n{content}\n"
        )

    def add_table(
        self,
        title: str,
        dataframe: pd.DataFrame,
        filename: str,
        note: str = None
    ):
        """
        Сохраняет таблицу в CSV и добавляет её в отчёт.
        """
        table_path = self.run_dir / filename
        dataframe.to_csv(
            table_path,
            index=False,
            encoding="utf-8-sig"
        )

        markdown_table = dataframe.to_markdown(index=False)

        content = markdown_table

        if note:
            content += f"\n\n{note}"

        content += (
            f"\n\nФайл таблицы: `{filename}`"
        )

        self.add_section(title, content)

    def add_figure(
        self,
        title: str,
        filename: str,
        description: str
    ):
        """
        Добавляет график в Markdown-отчёт.
        """
        relative_path = f"figures/{filename}"

        content = (
            f"![{title}]({relative_path})\n\n"
            f"*{title}. {description}*"
        )

        self.add_section(title, content)

    def add_optimization_result(
        self,
        result: dict,
        best: dict
    ):
        """
        Добавляет результаты обратной задачи в отчёт.
        """
        content = []

        content.append(
            "**Требуемая вероятность поражения:** "
            f"{result.get('p_required', '—')}"
        )

        content.append(
            "**Рекомендуемое количество НАР с кассетной боевой частью:** "
            f"{best.get('N', '—')} шт."
        )

        content.append(
            "**Оптимальная высота пуска:** "
            f"{best.get('H', '—')} м"
        )

        content.append(
            "**Оптимальная дальность до цели:** "
            f"{best.get('D', '—')} м"
        )

        content.append(
            "**Оптимальная скорость носителя:** "
            f"{best.get('V', '—')} м/с"
        )

        content.append(
            "**Прогнозируемая вероятность поражения:** "
            f"{best.get('P_kill', '—')}"
        )

        self.add_section(
            "Результаты решения обратной задачи",
            "\n\n".join(content)
        )

    def save_report(
        self,
        run_info: dict = None
    ) -> Path:
        """
        Сохраняет итоговый Markdown-отчёт.
        """
        timestamp = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

        header = [
            "# Отчёт о выполнении программной системы",
            "",
            f"**Дата и время формирования отчёта:** {timestamp}",
            ""
        ]

        if run_info:
            for key, value in run_info.items():
                header.append(f"- **{key}:** {value}")

            header.append("")

        full_report = "\n".join(header)
        full_report += "\n".join(self.report_sections)

        report_path = self.run_dir / "report.md"

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as file:
            file.write(full_report)

        print(f"Отчёт о выполнении сохранён: {report_path}")

        return report_path


print("Модуль application_layer.services.report_service определён.")