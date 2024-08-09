"""Torch Module for SubgraphX"""

""" Adapted from DGL Library"""

import math
from collections import Counter

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from dgl import to_heterogeneous, to_homogeneous
from dgl.base import NID
from dgl.convert import to_networkx
from dgl.subgraph import khop_in_subgraph, node_subgraph
from dgl.transforms.functional import remove_nodes

__all__ = ["SubgraphX", "HeteroSubgraphX"]


class MCTSNode:
    r"""Monte Carlo Tree Search Node

    Parameters
    ----------
    nodes : Tensor
        The node IDs of the graph that are associated with this tree node
    """

    def __init__(self, nodes):
        self.nodes = nodes
        self.num_visit = 0
        self.total_reward = 0.0
        self.immediate_reward = 0.0
        self.children = []

    def __repr__(self):
        r"""Get the string representation of the node.

        Returns
        -------
        str
            The string representation of the node
        """
        return str(self.nodes)


class NodeSubgraphX(nn.Module):
    r"""SubgraphX from `On Explainability of Graph Neural Networks via Subgraph
    Explorations <https://arxiv.org/abs/2102.05152>`__, adapted for heterogeneous graphs

    It identifies the most important subgraph from the original graph that
    plays a critical role in GNN-based graph classification.

    It employs Monte Carlo tree search (MCTS) in efficiently exploring
    different subgraphs for explanation and uses Shapley values as the measure
    of subgraph importance.

    Parameters
    ----------
    model : nn.Module
        The GNN model to explain that tackles multiclass graph classification

        * Its forward function must have the form
          :attr:`forward(self, graph, nfeat)`.
        * The output of its forward function is the logits.
    num_hops : int
        Number of message passing layers in the model
    coef : float, optional
        This hyperparameter controls the trade-off between exploration and
        exploitation. A higher value encourages the algorithm to explore
        relatively unvisited nodes. Default: 10.0
    high2low : bool, optional
        If True, it will use the "High2low" strategy for pruning actions,
        expanding children nodes from high degree to low degree when extending
        the children nodes in the search tree. Otherwise, it will use the
        "Low2high" strategy. Default: True
    num_child : int, optional
        This is the number of children nodes to expand when extending the
        children nodes in the search tree. Default: 12
    num_rollouts : int, optional
        This is the number of rollouts for MCTS. Default: 20
    node_min : int, optional
        This is the threshold to define a leaf node based on the number of
        nodes in a subgraph. Default: 3
    shapley_steps : int, optional
        This is the number of steps for Monte Carlo sampling in estimating
        Shapley values. Default: 100
    log : bool, optional
        If True, it will log the progress. Default: False
    """

    def __init__(
        self,
        model,
        num_hops,
        coef=10.0,
        high2low=True,
        num_child=12,
        num_rollouts=10,
        node_min=1,
        shapley_steps=20,
        log=True,
    ):
        super().__init__()
        self.num_hops = num_hops
        self.coef = coef
        self.high2low = high2low
        self.num_child = num_child
        self.num_rollouts = num_rollouts
        self.node_min = node_min
        self.shapley_steps = shapley_steps
        self.log = log

        self.model = model

    def shapley(self, subgraph_nodes):
        r"""Compute Shapley value with Monte Carlo approximation.

        Parameters
        ----------
        subgraph_nodes : dict[str, Tensor]
            subgraph_nodes[nty] gives the tensor node IDs of node type nty
            in the subgraph, which are associated with this tree node

        Returns
        -------
        float
            Shapley value
        """
        # Obtain neighboring nodes of the subgraph g_i, P'.
        local_regions = {
            ntype: nodes.tolist() for ntype, nodes in subgraph_nodes.items()
        }
        for _ in range(self.num_hops - 1):
            for c_etype in self.graph.canonical_etypes:
                src_ntype, _, dst_ntype = c_etype
                if src_ntype not in local_regions or dst_ntype not in local_regions:
                    continue

                in_neighbors, _ = self.graph.in_edges(
                    local_regions[dst_ntype], etype=c_etype
                )
                _, out_neighbors = self.graph.out_edges(
                    local_regions[src_ntype], etype=c_etype
                )
                local_regions[src_ntype] = list(
                    set(local_regions[src_ntype] + in_neighbors.tolist())
                )
                local_regions[dst_ntype] = list(
                    set(local_regions[dst_ntype] + out_neighbors.tolist())
                )

        split_point = self.graph.num_nodes()
        coalition_space = {
            ntype: list(set(local_regions[ntype]) - set(subgraph_nodes[ntype].tolist()))
            + [split_point]
            for ntype in subgraph_nodes.keys()
        }

        marginal_contributions = []
        for _ in range(self.shapley_steps):
            selected_node_map = dict()
            for ntype, nodes in coalition_space.items():
                permuted_space = np.random.permutation(nodes)
                split_idx = int(np.where(permuted_space == split_point)[0])
                selected_node_map[ntype] = permuted_space[:split_idx]

            # Mask for coalition set S_i
            exclude_mask = {
                ntype: torch.ones(self.graph.num_nodes(ntype))
                for ntype in self.graph.ntypes
            }
            for ntype, region in local_regions.items():
                exclude_mask[ntype][region] = 0.0
            for ntype, selected_nodes in selected_node_map.items():
                exclude_mask[ntype][selected_nodes] = 1.0

            # Mask for set S_i and g_i
            include_mask = {
                ntype: exclude_mask[ntype].clone() for ntype in self.graph.ntypes
            }
            for ntype, subgn in subgraph_nodes.items():
                exclude_mask[ntype][subgn] = 1.0

            ################################
            ################################
            # modified from DGL implementation
            # begin
            exclude_feat = {
                ntype: self.feat[ntype]
                * exclude_mask[ntype].unsqueeze(1).to(self.feat[ntype].device)
                for ntype in self.graph.ntypes
            }
            include_feat = {
                ntype: self.feat[ntype]
                * include_mask[ntype].unsqueeze(1).to(self.feat[ntype].device)
                for ntype in self.graph.ntypes
            }

            with torch.no_grad():
                exclude_probs = self.model(self.graph, exclude_feat, **self.kwargs)[
                    self.category
                ]
                include_probs = self.model(self.graph, include_feat, **self.kwargs)[
                    self.category
                ]
            marginal_contributions.append(include_probs - exclude_probs)

        return torch.cat(marginal_contributions).mean().item()
        # modify end
        ################################
        ################################

    def get_mcts_children(self, mcts_node):
        r"""Get the children of the MCTS node for the search.

        Parameters
        ----------
        mcts_node : MCTSNode
            Node in MCTS

        Returns
        -------
        list
            Children nodes after pruning
        """
        if len(mcts_node.children) > 0:
            return mcts_node.children

        subg = node_subgraph(self.graph, mcts_node.nodes)
        # Choose k nodes based on the highest degree in the subgraph
        node_degrees_map = {
            ntype: torch.zeros(subg.num_nodes(ntype), device=subg.nodes(ntype).device)
            for ntype in subg.ntypes
        }
        for c_etype in subg.canonical_etypes:
            src_ntype, _, dst_ntype = c_etype
            node_degrees_map[src_ntype] += subg.out_degrees(etype=c_etype)
            node_degrees_map[dst_ntype] += subg.in_degrees(etype=c_etype)

        node_degrees_list = [
            ((ntype, i), degree)
            for ntype, node_degrees in node_degrees_map.items()
            for i, degree in enumerate(node_degrees)
        ]
        node_degrees = torch.stack([v for _, v in node_degrees_list])
        k = min(subg.num_nodes(), self.num_child)
        chosen_node_indicies = torch.topk(
            node_degrees, k, largest=self.high2low
        ).indices
        chosen_nodes = [node_degrees_list[i][0] for i in chosen_node_indicies]

        mcts_children_maps = dict()

        for ntype, node in chosen_nodes:
            new_subg = remove_nodes(subg, node, ntype, store_ids=True)

            if new_subg.num_edges() > 0:
                new_subg_homo = to_homogeneous(new_subg)
                # Get the largest weakly connected component in the subgraph.
                nx_graph = to_networkx(new_subg_homo.cpu())
                largest_cc_nids = list(
                    max(nx.weakly_connected_components(nx_graph), key=len)
                )
                largest_cc_homo = node_subgraph(new_subg_homo, largest_cc_nids)
                largest_cc_hetero = to_heterogeneous(
                    largest_cc_homo, new_subg.ntypes, new_subg.etypes
                )

                # Follow steps for backtracking to original graph node ids
                # 1. retrieve instanced homograph from connected-component homograph
                # 2. retrieve instanced heterograph from instanced homograph
                # 3. retrieve hetero-subgraph from instanced heterograph
                # 4. retrieve orignal graph ids from subgraph node ids
                cc_nodes = {
                    ntype: subg.ndata[NID][ntype][
                        new_subg.ndata[NID][ntype][
                            new_subg_homo.ndata[NID][
                                largest_cc_homo.ndata[NID][indicies]
                            ]
                        ]
                    ]
                    for ntype, indicies in largest_cc_hetero.ndata[NID].items()
                }
            else:
                available_ntypes = [
                    ntype for ntype in new_subg.ntypes if new_subg.num_nodes(ntype) > 0
                ]
                chosen_ntype = np.random.choice(available_ntypes)
                # backtrack from subgraph node ids to entire graph
                chosen_node = subg.ndata[NID][chosen_ntype][
                    np.random.choice(
                        new_subg.nodes[chosen_ntype].data[NID].cpu().numpy()
                    )
                ]
                cc_nodes = {
                    chosen_ntype: torch.tensor(
                        [chosen_node],
                        device=subg.device,
                    )
                }

            if str(cc_nodes) not in self.mcts_node_maps:
                child_mcts_node = MCTSNode(cc_nodes)
                self.mcts_node_maps[str(child_mcts_node)] = child_mcts_node
            else:
                child_mcts_node = self.mcts_node_maps[str(cc_nodes)]

            if str(child_mcts_node) not in mcts_children_maps:
                mcts_children_maps[str(child_mcts_node)] = child_mcts_node

        mcts_node.children = list(mcts_children_maps.values())
        for child_mcts_node in mcts_node.children:
            if child_mcts_node.immediate_reward == 0:
                child_mcts_node.immediate_reward = self.shapley(child_mcts_node.nodes)

        return mcts_node.children

    def mcts_rollout(self, mcts_node):
        r"""Perform a MCTS rollout.

        Parameters
        ----------
        mcts_node : MCTSNode
            Starting node for MCTS

        Returns
        -------
        float
            Reward for visiting the node this time
        """
        if sum(len(nodes) for nodes in mcts_node.nodes.values()) <= self.node_min:
            return mcts_node.immediate_reward

        children_nodes = self.get_mcts_children(mcts_node)
        children_visit_sum = sum([child.num_visit for child in children_nodes])
        children_visit_sum_sqrt = math.sqrt(children_visit_sum)
        chosen_child = max(
            children_nodes,
            key=lambda c: c.total_reward / max(c.num_visit, 1)
            + self.coef
            * c.immediate_reward
            * children_visit_sum_sqrt
            / (1 + c.num_visit),
        )
        reward = self.mcts_rollout(chosen_child)
        chosen_child.num_visit += 1
        chosen_child.total_reward += reward

        return reward

    def get_exp_as_graph(self, explanation):
        from dgl import node_subgraph

        result = {}
        for node, indices in explanation.items():
            values = []
            for index in indices:
                values.append(self.graph.ndata["_ID"][node][index])
            result[node] = values
        exp_as_graph = node_subgraph(self.comp_graph, result)
        return exp_as_graph

    def get_most_frequent(self, lst):
        counts = Counter(lst)
        sorted_list = sorted(lst, key=lambda x: counts[x], reverse=True)
        most_frequent = sorted_list[0]
        return most_frequent

    # Newly defined function
    def explain_node(self, graph, feat, node_idx, category, **kwargs):
        """
        Executes an explanation process on a given node of a graph using Monte Carlo Tree Search (MCTS).

        This method takes a graph and a specific node index, and it applies MCTS to identify a subgraph that contributes to the model's prediction for that node.
        The method evaluates the model, extracts a k-hop subgraph around the target node, and iteratively explores subgraphs using MCTS to find the most explanatory subgraph.

        Parameters:
            category (str): The category of the node to be explained.
            node_idx (int): The index of the node in the graph for which the explanation is generated.
            graph (DGLGraph): The entire graph on which the model is trained.
            feat (dict): A dictionary containing features for each node type in the graph.
            **kwargs: Additional keyword arguments.

        Returns:
            exp_as_graph (DGLGraph): The explanatory subgraph identified by the method.
            predictions (int or list or -1): The predicted label(s) for the target node based on the explanatory subgraph. Returns -1 if no valid prediction could be made.

        Raises:
            AssertionError: If the number of nodes in the subgraph is smaller than a predefined minimum.

        Note:
            - The method uses the provided graph neural network model for predictions.
            - It operates under the assumption that the model and the necessary components (like MCTSNode) are already defined in the class.

        Example:
            >>> explainer = SomeExplainerClass(model)
            >>> explanatory_subgraph, prediction = explainer.some_method('category', node_index, graph, features)
            # This will generate an explanation for the specified node in the form of a subgraph and its prediction.
        """

        self.model.eval()
        self.category = category
        self.node_idx = node_idx
        self.comp_graph = graph
        exp_graph, _ = khop_in_subgraph(graph, {self.category: self.node_idx}, 1)
        assert (
            exp_graph.num_nodes() > self.node_min
        ), f"The number of nodes in the\
            graph {graph.num_nodes()} should be bigger than {self.node_min}."

        self.graph = exp_graph
        sg_nodes = self.graph.ndata[NID]
        sg_feat = {}

        for node_type in sg_nodes.keys():
            sg_feat[node_type] = feat[node_type][sg_nodes[node_type].long()]
        self.feat = sg_feat
        self.kwargs = kwargs

        # book all nodes in MCTS
        self.mcts_node_maps = dict()
        root_dict = {ntype: self.graph.nodes(ntype) for ntype in self.graph.ntypes}
        root = MCTSNode(root_dict)
        self.mcts_node_maps[str(root)] = root

        for i in range(self.num_rollouts):
            if self.log:
                print(
                    f"Rollout {i}/{self.num_rollouts}, \
                    {len(self.mcts_node_maps)} subgraphs have been explored."
                )
            self.mcts_rollout(root)

        best_leaf = None
        best_leaf_with_index = None
        best_immediate_reward = float("-inf")

        for mcts_node in self.mcts_node_maps.values():
            total_nodes = sum(tensor.numel() for tensor in mcts_node.nodes.values())
            if total_nodes < self.node_min:
                continue

            if mcts_node.immediate_reward > best_immediate_reward:
                best_leaf = mcts_node
                best_immediate_reward = mcts_node.immediate_reward

                # Check for index condition
                temp = self.get_exp_as_graph(best_leaf.nodes)
                node_mapping = temp.ndata[NID][category]
                index = self.node_idx in node_mapping
                if index:
                    best_leaf_with_index = mcts_node
                    best_immediate_reward = mcts_node.immediate_reward

        if best_leaf is None or best_leaf.immediate_reward == float("-inf"):
            return None, -1

        if best_leaf_with_index is None:
            exp_as_graph = temp
            sg_nodes = exp_as_graph.ndata[NID]
            sg_feat = {}
            for node_type in sg_nodes.keys():
                sg_feat[node_type] = feat[node_type][sg_nodes[node_type].long()]
            pred_logits = self.model(exp_as_graph, sg_feat)
            lst = pred_logits[category].argmax(dim=1).tolist()
            if len(lst) == 0:
                return exp_as_graph, -1
            predictions = self.get_most_frequent(lst)
        else:
            exp_as_graph = self.get_exp_as_graph(best_leaf_with_index.nodes)
            sg_nodes = exp_as_graph.ndata[NID]
            sg_feat = {}
            for node_type in sg_nodes.keys():
                sg_feat[node_type] = feat[node_type][sg_nodes[node_type].long()]
            pred_logits = self.model(exp_as_graph, sg_feat)[category]
            node_mapping = exp_as_graph.ndata[NID][category]
            index = (node_mapping == self.node_idx).nonzero().item()
            if len(pred_logits.shape) == 1:
                pred_logits = pred_logits.unsqueeze(0)
            predictions = pred_logits.argmax(dim=1).tolist()[index]
        return exp_as_graph, predictions
