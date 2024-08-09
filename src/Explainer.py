import json
import os
import time

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch as th
import torch.nn.functional as F
import yaml
from torch import nn

torch.cuda.empty_cache()

from dgl.nn.pytorch.explain import HeteroPGExplainer

from src.dglnn_local.subgraphx import NodeSubgraphX
from src.gnn_model.configs import get_configs
from src.gnn_model.dataset import RDFDatasets
from src.gnn_model.GAT import RGAT
from src.gnn_model.hetero_features import HeteroFeature
from src.gnn_model.RGCN import RGCN
from src.gnn_model.utils import (calculate_metrics, gen_evaluations,
                                 get_lp_aifb_fid, get_lp_bgs_fid,
                                 get_lp_mutag_fid, get_nodes_dict)
from src.logical_explainers.CELOE import train_celoe
from src.logical_explainers.EvoLearner import train_evo


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.best_model = model
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.best_model = model
        return self.early_stop


class Explainer:
    def __init__(
        self,
        explainers: list,
        dataset: str,
        model_name: str = "RGCN",
    ):
        self.explainers = explainers
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        self.configs = get_configs(self.dataset, model=self.model_name)
        self.hidden_dim = self.configs["hidden_dim"]
        self.num_bases = self.configs["n_bases"]
        self.lr = self.configs["lr"]
        self.weight_decay = self.configs["weight_decay"]
        self.epochs = self.configs["max_epoch"]
        self.validation = self.configs["validation"]
        self.hidden_layers = self.configs["num_layers"] - 1
        self.act = None
        self.patience = self.configs["patience"]

        self.my_dataset = RDFDatasets(
            self.dataset, root="data/", validation=self.validation
        )

        self.g = self.my_dataset.g.to(self.device)
        self.out_dim = self.my_dataset.num_classes
        self.e_types = self.g.etypes
        self.category = self.my_dataset.category
        self.train_idx = self.my_dataset.train_idx.to(self.device)
        self.test_idx = self.my_dataset.test_idx.to(self.device)
        self.labels = self.my_dataset.labels.to(self.device)
        self.rel_names = list(set(self.e_types))

        if self.validation:
            self.valid_idx = self.my_dataset.valid_idx.to(self.device)
        else:
            self.valid_idx = self.my_dataset.test_idx.to(self.device)

        self.idx_map = self.my_dataset.idx_map
        self.pred_df = pd.DataFrame(
            [
                {"IRI": self.idx_map[idx]["IRI"], "idx": idx}
                for idx in self.test_idx.tolist()
            ]
        )

        self.dataset_function_mapping = {
            "mutag": get_lp_mutag_fid,
            "aifb": get_lp_aifb_fid,
            "bgs": get_lp_bgs_fid,
            # Add more dataset-function mappings as needed
        }

        self.time_traker = {}
        self.explanations = {}
        self.evaluations = {}

        self.input_feature = HeteroFeature(
            {}, get_nodes_dict(self.g), self.hidden_dim, act=self.act
        ).to(self.device)

        if self.model_name == "RGCN":
            print("Initializing RGCN  model")
            self.model = RGCN(
                self.hidden_dim,
                self.hidden_dim,
                self.out_dim,
                self.e_types,
                self.num_bases,
                self.category,
                num_hidden_layers=self.hidden_layers,
            ).to(self.device)

        if self.model_name == "RGAT":
            if self.dataset == "mutag":
                self.act = F.elu
            print("Initializing R-GAT  model")
            self.model = RGAT(
                self.hidden_dim,
                self.out_dim,
                self.hidden_dim,
                self.e_types,
                num_heads=3,
                num_hidden_layers=self.hidden_layers,
            ).to(self.device)
        self.optimizer = th.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.loss_fn = F.cross_entropy
        self.model.add_module("input_feature", self.input_feature)
        self.optimizer.add_param_group({"params": self.input_feature.parameters()})
        self.early_stopping = EarlyStopping(patience=self.patience)
        self.feat = self.model.input_feature()

        self.train()
        self.run_explainers()

    def train(self):
        print("Start training...")
        dur = []
        train_accs = []
        val_accs = []
        vald_loss = []
        self.model.train()
        for epoch in range(self.epochs):
            t0 = time.time()
            logits = self.model(self.g, self.feat)[self.category]
            loss = self.loss_fn(logits[self.train_idx], self.labels[self.train_idx])
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            t1 = time.time()

            dur.append(t1 - t0)
            train_acc = th.sum(
                logits[self.train_idx].argmax(dim=1) == self.labels[self.train_idx]
            ).item() / len(self.train_idx)
            train_accs.append(train_acc)

            val_loss = F.cross_entropy(
                logits[self.valid_idx], self.labels[self.valid_idx]
            )
            val_acc = th.sum(
                logits[self.valid_idx].argmax(dim=1) == self.labels[self.valid_idx]
            ).item() / len(self.valid_idx)
            val_accs.append(val_acc)
            vald_loss.append(val_loss.item())
            print(
                "Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Valid Acc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".format(
                    epoch,
                    train_acc,
                    loss.item(),
                    val_acc,
                    val_loss.item(),
                    np.average(dur),
                )
            )
            if self.early_stopping.step(val_acc, self.model):
                print("Early stopping")

                break
        print("End Training")
        self.model = self.early_stopping.best_model
        pred_logit = self.model(self.g, self.feat)[self.category]

        val_acc_final = th.sum(
            pred_logit[self.test_idx].argmax(dim=1) == self.labels[self.test_idx]
        ).item() / len(self.test_idx)
        print(
            f"Final validation accuracy of the model {self.model_name} on unseen dataset: {val_acc_final}"
        )

        if self.validation:
            self.total_train_idx = torch.cat([self.train_idx, self.valid_idx], dim=0)
        else:
            self.total_train_idx = self.train_idx

        # Train data
        self.gnn_pred_dt_train = {
            t_idx: pred
            for t_idx, pred in zip(
                self.total_train_idx.tolist(),
                pred_logit[self.total_train_idx].argmax(dim=1).tolist(),
            )
        }
        self.gt_dict_train = {
            t_idx: pred
            for t_idx, pred in zip(
                self.total_train_idx.tolist(),
                self.labels[self.total_train_idx].tolist(),
            )
        }

        # Test data
        self.gnn_pred_dt_test = {
            t_idx: pred
            for t_idx, pred in zip(
                self.test_idx.tolist(), pred_logit[self.test_idx].argmax(dim=1).tolist()
            )
        }
        self.gt_dict_test = {
            t_idx: pred
            for t_idx, pred in zip(
                self.test_idx.tolist(), self.labels[self.test_idx].tolist()
            )
        }

        self.pred_df["Ground Truths"] = self.pred_df["idx"].map(self.gt_dict_test)
        self.pred_df["GNN Preds"] = self.pred_df["idx"].map(self.gnn_pred_dt_test)
        self.create_lp()

    def _run_evo(self):
        # pred ability of EvoLearner can be computed using ground truth LPs.
        # uncomment the code below and save the predictions into a new column in pred_df
        # predicition w.r.t Ground truths can examine the capability of EvoLearner as independent node classifier
        # print(f"Training EvoLearner (prediction) on {self.dataset}")
        # target_dict_evo_pred, duration_evo_pred, _ = train_evo(
        #     learning_problems=self.learning_problem_pred, kg=self.dataset
        # )
        # print(
        #     f"Total time taken for EvoLearner (prediction)  on {self.dataset}: {duration_evo_pred:.2f}"
        # )

        print(f"Training EvoLearner (explanation) on {self.dataset}")
        target_dict_evo_exp, duration_evo_exp, explanation_dict_evo = train_evo(
            learning_problems=self.learning_problem_fid, kg=self.dataset
        )
        print(
            f"Total time taken for EvoLearner (explanation)  on {self.dataset}: {duration_evo_exp:.2f}"
        )

        self.time_traker["EvoLearner"] = duration_evo_exp
        # self.pred_df["Evo(pred)"] = self.pred_df["IRI"].map(target_dict_evo_pred)
        self.pred_df["EvoLearner"] = self.pred_df["IRI"].map(target_dict_evo_exp)
        self.explanations["EvoLearner"] = explanation_dict_evo
        prediction_performance = calculate_metrics(
            self.pred_df["Ground Truths"], self.pred_df["EvoLearner"]
        )
        explanation_performance = calculate_metrics(
            self.pred_df["GNN Preds"], self.pred_df["EvoLearner"]
        )
        celoe_performance = gen_evaluations(
            prediction_performance=prediction_performance,
            explanation_performance=explanation_performance,
        )
        self.evaluations["EvoLearner"] = celoe_performance

    def _run_celoe(self):
        # pred ability of CELOE can be computed using ground truth LPs.
        # uncomment the code below and save the predictions into a new column in pred_df
        # predicition w.r.t Ground truths can examine the capability of CELOE as independent node classifier
        # print(f"Training CELOE (prediction) on {self.dataset}")
        # target_dict_celoe_pred, duration_celoe_pred, _ = train_celoe(
        #     learning_problems=self.learning_problem_pred, kg=self.dataset
        # )
        # print(
        #     f"Total time taken for CELOE (prediction)  on {self.dataset}: {duration_celoe_pred:.2f}"
        # )

        print(f"Training CELOE (explanation) on {self.dataset}")
        target_dict_celoe_exp, duration_celoe_exp, explanation_dict_celoe = train_celoe(
            learning_problems=self.learning_problem_fid, kg=self.dataset
        )
        print(
            f"Total time taken for CELOE (explanation)  on {self.dataset}: {duration_celoe_exp:.2f}"
        )

        self.time_traker["CELOE"] = duration_celoe_exp
        # self.pred_df["CELOE(pred)"] = self.pred_df["IRI"].map(target_dict_celoe_pred)
        self.pred_df["CELOE"] = self.pred_df["IRI"].map(target_dict_celoe_exp)
        self.explanations["CELOE"] = explanation_dict_celoe

        prediction_performance = calculate_metrics(
            self.pred_df["Ground Truths"], self.pred_df["CELOE"]
        )
        explanation_performance = calculate_metrics(
            self.pred_df["GNN Preds"], self.pred_df["CELOE"]
        )
        celoe_performance = gen_evaluations(
            prediction_performance=prediction_performance,
            explanation_performance=explanation_performance,
        )
        self.evaluations["CELOE"] = celoe_performance

    def _run_pgexplainer(self, print_explainer_loss=True):

        # Load configurations from the YAML file
        config_path = "configs/pgexplainer.yaml"
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        pg_config = config_data[self.dataset]
        init_tmp = pg_config["initial_temp"]
        final_tmp = pg_config["final_temp"]

        print("Starting PGExplainer")
        t0 = time.time()
        explainer_pg = HeteroPGExplainer(
            self.model, self.hidden_dim, num_hops=1, explain_graph=False
        )
        feat_pg = {item: self.feat[item].data for item in self.feat}

        optimizer_exp = th.optim.Adam(
            explainer_pg.parameters(), lr=pg_config["learning_rate"]
        )
        for epoch in range(pg_config["num_epochs"]):
            tmp = float(init_tmp * np.power(final_tmp / init_tmp, epoch / 20))
            loss = explainer_pg.train_step_node(
                {
                    ntype: self.g.nodes(ntype)
                    for ntype in self.g.ntypes
                    if ntype == self.category
                },
                self.g,
                feat_pg,
                tmp,
            )
            if loss < 0.5:
                print("Stopping Explainer due to very small loss")
                break
            optimizer_exp.zero_grad()
            loss.backward()
            optimizer_exp.step()
            if print_explainer_loss:
                print(f"Explainer trained for {epoch+1} epochs with loss {loss :3f}")
        self.entity_pg = {}
        probs, edge_mask, bg, inverse_indices = explainer_pg.explain_node(
            {self.category: self.test_idx}, self.g, self.feat, training=True
        )
        exp_pred_pg = (
            probs[self.category][inverse_indices[self.category]].argmax(dim=1).tolist()
        )
        t1 = time.time()
        pg_preds = dict(zip(self.test_idx.tolist(), exp_pred_pg))
        self.pred_df["PGExplainer"] = self.pred_df["idx"].map(pg_preds)

        prediction_performance = calculate_metrics(
            self.pred_df["Ground Truths"], self.pred_df["PGExplainer"]
        )
        explanation_performance = calculate_metrics(
            self.pred_df["GNN Preds"], self.pred_df["PGExplainer"]
        )
        pg_performance = gen_evaluations(
            prediction_performance=prediction_performance,
            explanation_performance=explanation_performance,
        )
        self.evaluations["PGExplainer"] = pg_performance

        dur_pg = t1 - t0
        self.time_traker["PGExplainer"] = dur_pg
        print(f"Total time taken for PGExplainer  on {self.dataset}: {dur_pg:.2f}")

    def _run_subgraphx(self):

        print("Starting SubGraphX")
        t0 = time.time()
        explainer_sgx = NodeSubgraphX(
            self.model, num_hops=1, num_rollouts=3, shapley_steps=5
        )
        exp_preds_sgx = {}
        for idx in self.test_idx.tolist():
            print(idx)
            explanation, prediction = explainer_sgx.explain_node(
                self.g, self.feat, idx, self.category
            )
            exp_preds_sgx[idx] = prediction

        t1 = time.time()
        self.pred_df["SubGraphX"] = self.pred_df["idx"].map(exp_preds_sgx)
        dur_sgx = t1 - t0
        self.time_traker["SubGraphX"] = dur_sgx

        prediction_performance = calculate_metrics(
            self.pred_df["Ground Truths"], self.pred_df["SubGraphX"]
        )
        explanation_performance = calculate_metrics(
            self.pred_df["GNN Preds"], self.pred_df["SubGraphX"]
        )
        sgx_performance = gen_evaluations(
            prediction_performance=prediction_performance,
            explanation_performance=explanation_performance,
        )
        self.evaluations["SubGraphX"] = sgx_performance

        print(f"Total time taken for SubGraphX  on {self.dataset}: {dur_sgx:.2f}")

    def run_explainers(self):
        explainer_methods = {
            "EvoLearner": self._run_evo,
            "CELOE": self._run_celoe,
            "PGExplainer": self._run_pgexplainer,
            "SubGraphX": self._run_subgraphx,
            # Add more explainer-method mappings as needed
        }
        if self.explainers is not None:
            for explainer in self.explainers:
                explainer_method = explainer_methods.get(explainer)
                if explainer_method:
                    explainer_method()

    def create_lp(self):
        self.lp_function = self.dataset_function_mapping.get(self.dataset)

        if self.lp_function:
            self.learning_problem_pred = self.lp_function(
                self.gt_dict_train, self.gt_dict_test, self.idx_map
            )

            self.learning_problem_fid = self.lp_function(
                self.gnn_pred_dt_train, self.gnn_pred_dt_test, self.idx_map
            )
