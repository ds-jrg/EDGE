import json
import os
from statistics import mean

from prettytable import PrettyTable


def print_results(model_name="RGCN"):
    # Initialize a PrettyTable
    table = PrettyTable()
    table.field_names = [
        "Model",
        "Dataset",
        "Pred Accuracy",
        "Pred Precision",
        "Pred Recall",
        "Pred F1 Score",
        "Exp Accuracy",
        "Exp Precision",
        "Exp Recall",
        "Exp F1 Score",
    ]

    models = ["CELOE", "EvoLearner", "PGExplainer", "SubGraphX"]
    datasets = ["aifb", "mutag", "bgs"]
    for model in models:
        for dataset in datasets:
            file_path = f"results/evaluations/{model_name}/{model}/{dataset}.json"
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                continue

            if not data:
                continue

            pred_accuracy = round(
                mean([metrics["pred_accuracy"] for _, metrics in data.items()]), 3
            )
            pred_precision = round(
                mean([metrics["pred_precision"] for _, metrics in data.items()]), 3
            )
            pred_recall = round(
                mean([metrics["pred_recall"] for _, metrics in data.items()]), 3
            )
            pred_f1_score = round(
                mean([metrics["pred_f1_score"] for _, metrics in data.items()]), 3
            )
            exp_accuracy = round(
                mean([metrics["exp_accuracy"] for _, metrics in data.items()]), 3
            )
            exp_precision = round(
                mean([metrics["exp_precision"] for _, metrics in data.items()]), 3
            )
            exp_recall = round(
                mean([metrics["exp_recall"] for _, metrics in data.items()]), 3
            )
            exp_f1_score = round(
                mean([metrics["exp_f1_score"] for _, metrics in data.items()]), 3
            )

            # Add the results to the PrettyTable
            table.add_row(
                [
                    model,
                    dataset,
                    pred_accuracy,
                    pred_precision,
                    pred_recall,
                    pred_f1_score,
                    exp_accuracy,
                    exp_precision,
                    exp_recall,
                    exp_f1_score,
                ]
            )
    table.sortby = "Dataset"
    print(table)
