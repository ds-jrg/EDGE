""" Code Adapted from https://github.com/BUPT-GAMMA/OpenHGNN"""

import dgl
import dgl.function as fn
import numpy as np
import torch as th

from src.dglnn_local.RDFDataset import (AIFBDataset, AMDataset, BGSDataset,
                                        MUTAGDataset)


class RDFDatasets:
    """
    A class for handling RDF datasets, specifically designed for graph-based machine learning models.

    This class is responsible for loading and preparing RDF datasets, such as AIFB, MUTAG, BGS, and AM, for use in graph neural network (GNN) models.
    It supports various dataset operations, including loading graphs, handling labels, and creating train/test/validation splits.

    Attributes:
        g (DGLGraph): A DGL graph object.
        category (str): The target category for prediction in the dataset.
        num_classes (int): The number of classes in the dataset.
        has_feature (bool): Indicates whether the nodes in the graph have features.
        multi_label (bool): Indicates whether the dataset is a multi-label classification problem.
        meta_paths_dict (dict): A dictionary containing meta-paths for the graph (if applicable).
        idx_map (dict): A mapping from node indices to their corresponding labels.
        labels (Tensor): The labels for the nodes in the graph.
        train_idx (Tensor): Indices of the training set nodes.
        valid_idx (Tensor): Indices of the validation set nodes.
        test_idx (Tensor): Indices of the test set nodes.

    Parameters:
        dataset (str): The name of the dataset to load. Supported datasets are 'aifb', 'mutag', 'bgs', and 'am'.
        root (str, optional): The root directory where the dataset is stored. Defaults to None.
        validation (bool, optional): Indicates whether a validation set should be created. Defaults to True.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Raises:
        ValueError: If the specified dataset is not supported or if required data (like labels) is missing in the dataset.

    Example:
        >>> rdf_dataset = RDFDatasets(dataset="aifb", root="/path/to/dataset", validation=True)
        >>> print(rdf_dataset.g) # prints the DGLGraph object
    """

    def __init__(self, dataset, root=None, validation=True, *args, **kwargs):
        self.g = None
        self.category = None
        self.num_classes = None
        self.has_feature = False
        self.multi_label = False
        self.meta_paths_dict = None
        # self.in_dim = None

        # load graph data
        if dataset == "aifb":
            kg_dataset = AIFBDataset(raw_dir=root)
        elif dataset == "mutag":
            kg_dataset = MUTAGDataset(raw_dir=root)
        elif dataset == "bgs":
            kg_dataset = BGSDataset(raw_dir=root)
        elif dataset == "am":
            kg_dataset = AMDataset(raw_dir=root)
        else:
            raise ValueError()

        # Load from hetero-graph
        self.g = kg_dataset[0]
        self.category = kg_dataset.predict_category
        self.num_classes = kg_dataset.num_classes
        self.idx_map = kg_dataset.idx_map

        if "labels" in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop("labels").long()
        elif "label" in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop("label").long()
        else:
            raise ValueError("Labels of nodes are not in the hg.nodes[category].data.")
        self.labels = labels.float() if self.multi_label else labels

        if "train_mask" not in self.g.nodes[self.category].data:
            # self.logger.dataset_info("The dataset has no train mask. "
            #       "So split the category nodes randomly. And the ratio of train/test is 8:2.")
            num_nodes = self.g.number_of_nodes(self.category)
            n_test = int(num_nodes * 0.2)
            n_train = num_nodes - n_test

            train, test = th.utils.data.random_split(
                range(num_nodes), [n_train, n_test]
            )
            train_idx = th.tensor(train.indices)
            test_idx = th.tensor(test.indices)
            if validation:
                # self.logger.dataset_info("Split train into train/valid with the ratio of 8:2 ")
                random_int = th.randperm(len(train_idx))
                valid_idx = train_idx[random_int[: len(train_idx) // 5]]
                train_idx = train_idx[random_int[len(train_idx) // 5 :]]
            else:
                # self.logger.dataset_info("Set valid set with train set.")
                valid_idx = train_idx
                train_idx = train_idx
        else:
            train_mask = self.g.nodes[self.category].data.pop("train_mask")
            test_mask = self.g.nodes[self.category].data.pop("test_mask")
            train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
            test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
            if validation:
                if "val_mask" in self.g.nodes[self.category].data:
                    val_mask = self.g.nodes[self.category].data.pop("val_mask")
                    valid_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
                elif "valid_mask" in self.g.nodes[self.category].data:
                    val_mask = (
                        self.g.nodes[self.category].data.pop("valid_mask").squeeze()
                    )
                    valid_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
                else:
                    # RDF_NodeClassification has train_mask, no val_mask
                    # self.logger.dataset_info("Split train into train/valid with the ratio of 8:2 ")
                    random_int = th.randperm(len(train_idx))
                    valid_idx = train_idx[random_int[: len(train_idx) // 5]]
                    train_idx = train_idx[random_int[len(train_idx) // 5 :]]
            else:
                # self.logger.dataset_info("Set valid set with train set.")
                valid_idx = train_idx
                train_idx = train_idx
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
