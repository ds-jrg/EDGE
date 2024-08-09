""" This example script demonstrates how to visualize the explanations produced by SubGraphX.
 Similar methods can be used to implement visualization for other GNN explainers.
 The dataset and model used can be varied."""

# The visulization result is stored in results/exp_visualizations

from src.dglnn_local.subgraphx import NodeSubgraphX
from src.Explainer import Explainer
from src.utils.visualize_hetero_graphs import visualize_hd

vis_dataset = "mutag"
vis_explainer = Explainer(explainers=[], dataset=vis_dataset, model_name="RGCN")
vis_category = vis_explainer.category
vis_model = vis_explainer.model
idx_map = vis_explainer.idx_map
node_id = vis_explainer.test_idx[0]
node_idx = node_id.tolist()
explainer = NodeSubgraphX(vis_model, num_hops=1, num_rollouts=3, shapley_steps=5)
feat = vis_model.input_feature()
explanation, logits = explainer.explain_node(
    vis_explainer.g, feat, node_id, vis_category
)
file_name = f"{vis_dataset}_subg"
visualize_hd(
    explanation,
    node_id=node_idx,
    file_name="exp_pg",
    edge_label_flag=True,
    caption=idx_map[node_idx]["IRI"],
)
