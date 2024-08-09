# Some part of Code adapted from https://github.com/mathematiger/ExplwCE/blob/master/visualization.py
import colorsys
import logging
import math
import os
import re

import dgl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import torch
from matplotlib.patches import Patch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("matplotlib")
logger.setLevel(logging.WARNING)


def generate_colors(num_colors):
    # Define the number of distinct hues to use
    num_hues = num_colors + 1
    # Generate a list of evenly spaced hues
    hues = [i / num_hues for i in range(num_hues)]
    # Shuffle the hues randomly
    # random.shuffle(hues)
    saturations = []
    # saturations = [0.8 for _ in range(num_colors)]
    values = []
    # values = [0.4 for _ in range(num_colors)]
    for i in range(num_colors):
        if i % 2 == 0:
            values.append(0.4)
            saturations.append(0.4)
        else:
            values.append(0.8)
            saturations.append(0.7)
    # Convert the hues, saturations, and values to RGB colors
    colors = [
        colorsys.hsv_to_rgb(h, s, v) for h, s, v in zip(hues, saturations, values)
    ]
    # Convert the RGB colors to hexadecimal strings
    hex_colors = [
        f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b in colors
    ]
    return hex_colors


def adjust_caption_size_exp(caption_length, max_size=18, min_size=8, rate=0.1):
    font_size = max_size * math.exp(-rate * caption_length)
    return max(min_size, min(font_size, max_size))


def visualize_hd(
    hd_graph,
    node_id=None,
    file_name=None,
    target_dir=None,
    caption=None,
    with_labels=True,
    edge_label_flag=False,
):
    """
    Visualizes a heterogenous graph using matplotlib and networkx.

    This function takes a heterogenous graph (from DGL library), and optionally a target node id,
    and produces a visualization of the graph. The visualization includes colored nodes and edges
    representing different types, and can optionally include labels and a caption.

    Parameters:
    hd_graph (DGLGraph): The heterogenous graph to be visualized.
    node_id (int, optional): The id of the target node to explain. Default is None.
    file_name (str, optional): Name of the file to save the visualization. Default is None.
    target_dir (str, optional): Directory where the visualization file will be saved. Default is None.
    caption (str, optional): Caption to be added to the visualization. Default is None.
    with_labels (bool, optional): Whether to include labels in the visualization. Default is True.
    edge_label_flag (bool, optional): Whether to include edge labels. Default is False.

    Returns:
    None: The function does not return anything but saves the visualization as a file and shows it.

    """
    try:
        plt.clf()
    except Exception as e:
        print(f"An exception occurred while clearing the plot: {e}")

    list_all_nodetypes = [
        ntype for ntype in hd_graph.ntypes if len(hd_graph.nodes(ntype)) > 0
    ]
    # create data for legend and caption
    curent_nodetypes_to_all_nodetypes = [
        [count, hd_graph.ntypes.index(_)] for count, _ in enumerate(list_all_nodetypes)
    ]

    number_of_node_types_for_colors = len(curent_nodetypes_to_all_nodetypes)
    colors = generate_colors(number_of_node_types_for_colors)

    # create nx graph to visualize
    Gnew = nx.Graph()
    homdata = dgl.to_homogeneous(hd_graph, edata=hd_graph.edata["_TPYE"])
    Gnew.add_nodes_from(list(range(homdata.num_nodes())))

    try:
        nodes_to_explain = [
            item
            for sublist in hd_graph.ndata[dgl.NID].values()
            for item in sublist.tolist()
        ].index(node_id)
    except:
        nodes_to_explain = -1
    if nodes_to_explain == -1:
        edge_label_flag = False

    label_dict = {}
    node_color = []
    ntypes_list = homdata.ndata["_TYPE"].tolist()
    for count, item in enumerate(ntypes_list):
        node_label_to_index = list_all_nodetypes.index(hd_graph.ntypes[item])
        label_dict[count] = list_all_nodetypes[node_label_to_index][:3]
        if count == nodes_to_explain:
            node_color.append("#6E4B4B")
        else:
            node_color.append(colors[node_label_to_index])

    # remove reverse edges as we plot a homogenous graph
    edge_types_ = [
        hd_graph.etypes[item].replace("rev-", "")
        if hd_graph.etypes[item].startswith("rev-")
        else hd_graph.etypes[item]
        for item in homdata.edata["_TYPE"].tolist()
    ]

    # creating edges with manual iteration to ensure original ordering of edges
    # torch.stack somehows changes the order in the edge list
    list_edges_for_networkx_ = [
        (a.item(), b.item()) for a, b in zip(homdata.edges()[0], homdata.edges()[1])
    ]

    list_edges_for_networkx = []
    edge_types = []
    seen = set()
    # filter reduntant edge types to map colors properly  as nx does that randomly
    for i, item in enumerate(list_edges_for_networkx_):
        if item not in seen:
            seen.add(item)
            list_edges_for_networkx.append(item)
            edge_types.append(edge_types_[i])
    edge_labels = [
        etype if nodes_to_explain in edge else "others"
        for edge, etype in zip(list_edges_for_networkx, edge_types)
    ]

    # if edge[0] == nodes_to_explain else "others"
    unique_edge_types = set(edge_labels)
    unique_edge_types_list = list(unique_edge_types)

    color_palette = plt.cm.get_cmap(
        "hsv", len(unique_edge_types)
    )  # Using HSV colormap for variety

    # Apply colors to edges based on type
    edge_type_color = {
        etype: "black"
        if etype == "others"
        else color_palette(unique_edge_types_list.index(etype))
        for etype in edge_labels
    }
    edge_color = [
        "black"
        if etype == "others"
        else color_palette(unique_edge_types_list.index(etype))
        for etype in edge_labels
    ]
    edges_with_colour = dict(zip(list_edges_for_networkx, edge_color))
    # Create a list of colors for each edge in the graph
    for edge, color in edges_with_colour.items():
        Gnew.add_edge(edge[0], edge[1], color=color)

    # plt
    pos = nx.kamada_kawai_layout(Gnew)
    options = {"with_labels": "True", "node_size": 500}
    nx.draw_networkx(
        Gnew,
        pos,
        node_color=node_color,
        **options,
        labels=label_dict,
    )

    if edge_label_flag:
        for edge, color in edges_with_colour.items():
            nx.draw_networkx_edges(Gnew, pos, edgelist=[edge], edge_color=[color])

    # create legend
    patch_list = []
    name_list = []
    for i in range(len(list_all_nodetypes)):
        patch_list.append(plt.Circle((0, 0), 0.1, fc=colors[i]))
        name_list.append(list_all_nodetypes[i])

    if nodes_to_explain != -1:
        special_node_color = "#6E4B4B"
        special_node_label = "Target Node"
        patch_list.append(Patch(color=special_node_color))
        name_list.append(special_node_label)

    name_list = [name[1:] if name[0] == "_" else name for name in name_list]

    # create caption
    if caption:
        if nodes_to_explain != -1:
            caption = "Explanation for Node ID : " + str(node_id) + "---> " + caption
        caption_text = caption
        caption_size = adjust_caption_size_exp(
            caption_length=len(caption), max_size=18, min_size=8, rate=0.1
        )
        caption_position = (0.5, 1.005)

    # folder to save in
    if target_dir:
        name_plot_save = f"{target_dir}/{file_name}"
    else:
        name_plot_save = f"results/exp_visualizations/{file_name}"
    directory = os.path.dirname(name_plot_save)
    os.makedirs(directory, exist_ok=True)

    if with_labels:
        # Create and place the legend for node colors

        node_legend = plt.legend(
            patch_list,
            name_list,
            title="Node Types",
            loc="upper left",
            bbox_to_anchor=(-0.01, -0.01),  # Adjust these values
            borderaxespad=0.0,
        )
        if edge_label_flag:
            # Create and place the legend for edge colors
            edge_patch_list = [
                plt.Line2D([0], [0], color=color, label=etype, linewidth=2)
                for etype, color in edge_type_color.items()
            ]
            # can use w/o assigning to the edge_legend variable
            edge_legend = plt.legend(
                handles=edge_patch_list,
                title="Edge Types",
                loc="upper right",
                bbox_to_anchor=(1.05, -0.01),
                borderaxespad=0.0,
            )
            plt.gca().add_artist(node_legend)

        # Define the file paths
        if node_id > 0:
            file_path_with_legend = f"{name_plot_save}_{node_id}.png"
        else:
            file_path_with_legend = f"{name_plot_save}.png"

        if caption:
            # Save the figure with legend and caption
            plt.figtext(*caption_position, caption_text, ha="center", size=caption_size)
        plt.tight_layout()
        plt.savefig(file_path_with_legend, bbox_inches="tight", dpi=200, format="png")

        # Show the plot
        plt.show()
    else:
        # Define the file paths
        file_path_wo_legend = f"{name_plot_save}_wo.png"
        # Save the figure without legend and caption
        plt.savefig(file_path_wo_legend, bbox_inches="tight")
        # plt.tight_layout()
        # Show the plot
        plt.show()
