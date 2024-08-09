import json
import os

import pandas as pd

from src.Explainer import Explainer


def run_explainers(
    dataset, explainers, print_explainer_loss=True, no_of_runs=5, model="RGCN"
):
    print(f"Running explainers for {no_of_runs} runs for dataset {dataset}")
    performances = {}
    preds = {}
    predictions_dfs = []

    for i in range(no_of_runs):
        print(f"Starting the  Run", i + 1)
        performances[i] = {}
        preds[i] = {}
        my_explainer = Explainer(
            explainers=explainers, dataset=dataset, model_name=model
        )
        for explainer in explainers:
            if explainer == "PGExplainer":
                performances[i][explainer] = my_explainer.evaluations.get(explainer, {})
            elif explainer == "SubGraphX":
                performances[i][explainer] = my_explainer.evaluations.get(explainer, {})
            elif explainer == "EvoLearner":
                performances[i][explainer] = my_explainer.evaluations.get(explainer, {})
                preds[i][explainer] = my_explainer.explanations.get(explainer, {})
            elif explainer == "CELOE":
                performances[i][explainer] = my_explainer.evaluations.get(explainer, {})
                preds[i][explainer] = my_explainer.explanations.get(explainer, {})
        my_explainer.pred_df["Run"] = int(i) + 1
        predictions_dfs.append(my_explainer.pred_df)

    for explainer in explainers:
        file_path_evaluations = (
            f"results/evaluations/{model}/{explainer}/{dataset}.json"
        )
        # Get the directory path from the file path
        directory = os.path.dirname(file_path_evaluations)
        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path_evaluations, "w") as json_file:
            json.dump(
                {f"Run_{i + 1}": performances[i][explainer] for i in range(no_of_runs)},
                json_file,
                indent=2,
            )

        if explainer not in ["PGExplainer", "SubGraphX"]:
            file_path_predictions = (
                f"results/predictions/{model}/{explainer}/{dataset}.json"
            )
            directory = os.path.dirname(file_path_predictions)
            # Create the directory if it doesn't exist
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(file_path_predictions, "w") as json_file:
                json.dump(
                    {f"Run_{i + 1}": preds[i][explainer] for i in range(no_of_runs)},
                    json_file,
                    indent=2,
                )
    big_df = pd.concat(predictions_dfs, ignore_index=True)
    file_path_predictions_df = f"results/predictions/{model}/{dataset}.csv"
    directory = os.path.dirname(file_path_predictions_df)
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    big_df.to_csv(file_path_predictions_df, index=False)
