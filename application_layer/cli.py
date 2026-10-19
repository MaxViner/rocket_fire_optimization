# ЯЧЕЙКА 11. МОДУЛЬ: application_layer/cli.py
#
# Назначение:
# - запуск сценариев из командной строки;
# - обучение моделей;
# - одиночный прогноз;
# - оптимизация параметров;
# - пакетная обработка.
# ============================================================
import argparse
import json

from config import CONFIG
from application_layer.api import ApplicationAPI


def build_cli():
    parser = argparse.ArgumentParser(
        description="CLI для модуля прогнозирования и оптимизации."
    )

    subparsers = parser.add_subparsers(dest="command")

    # --- Обучение ---
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data", type=str, default=None)
    train_parser.add_argument("--samples", type=int, default=5000)
    train_parser.add_argument("--tune", action="store_true")

    # --- Прогноз ---
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--input", type=str, required=True)
    predict_parser.add_argument("--H", type=float, required=True)
    predict_parser.add_argument("--D", type=float, required=True)
    predict_parser.add_argument("--V", type=float, required=True)
    predict_parser.add_argument("--N", type=int, required=True)
    predict_parser.add_argument("--C", type=str, required=True)
    predict_parser.add_argument("--T", type=str, required=True)
    predict_parser.add_argument("--W_dir", type=float, default=0)
    predict_parser.add_argument("--W_spd", type=float, default=0)
    predict_parser.add_argument("--Temp", type=float, default=0)

    # --- Оптимизация ---
    optimize_parser = subparsers.add_parser("optimize")
    optimize_parser.add_argument("--C", type=str, required=True)
    optimize_parser.add_argument("--T", type=str, required=True)
    optimize_parser.add_argument("--p_req", type=float, required=True)
    optimize_parser.add_argument("--W_dir", type=float, default=0)
    optimize_parser.add_argument("--W_spd", type=float, default=0)
    optimize_parser.add_argument("--Temp", type=float, default=0)

    # --- Пакетная обработка ---
    batch_parser = subparsers.add_parser("batch")
    batch_parser.add_argument("--input", type=str, required=True)
    batch_parser.add_argument("--output", type=str, required=True)

    return parser


def run_cli(args):
    api = ApplicationAPI(CONFIG)

    if args.command == "train":
        api.prepare_data(source=args.data, n_samples=args.samples)
        result = api.train_and_select_model(tune=args.tune)

        print(json.dumps(
            {
                "best_model": result["best_model"],
                "metrics": result["comparison"].to_dict(orient="records")
            },
            ensure_ascii=False,
            indent=4
        ))

    elif args.command == "predict":
        api.prepare_data()
        api.train_and_select_model()

        p = api.predict_probability({
            "H": args.H,
            "D": args.D,
            "V": args.V,
            "N": args.N,
            "C": args.C,
            "T": args.T,
            "W_dir": args.W_dir,
            "W_spd": args.W_spd,
            "Temp": args.Temp
        })

        print(f"Расчётная вероятность поражения: {p:.4f}")

    elif args.command == "optimize":
        api.prepare_data()
        api.train_and_select_model()

        result = api.optimize_mission(
            target_type=args.C,
            effect_type=args.T,
            weather={
                "dir": args.W_dir,
                "spd": args.W_spd,
                "temp": args.Temp
            },
            p_required=args.p_req
        )

        if result["found"]:
            print("Найдено решение:")
            print(json.dumps(result["best"], ensure_ascii=False, indent=4))
        else:
            print("Допустимое решение не найдено.")

    elif args.command == "batch":
        api.prepare_data()
        api.train_and_select_model()

        result = api.batch_predict(
            input_path=args.input,
            output_path=args.output
        )

        print(f"Обработано строк: {len(result)}")
        print(f"Результат сохранён: {args.output}")


# В Jupyter CLI обычно не запускают автоматически.
# Для терминала:
# python cli.py train --samples 5000 --tune
# python cli.py optimize --C stronghold --T destroy --p_req 0.85
print("Модуль application_layer.cli определён.")