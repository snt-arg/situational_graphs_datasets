from graph_datasets.SyntheticDatasetGenerator import SyntheticDatasetGenerator
from graph_datasets.graph_visualizer import visualize_nxgraph, visualize_nxgraph_3d
import matplotlib.pyplot as plt
import json, os, time, sys
import networkx as nx
from graph_wrapper.GraphWrapper import GraphWrapper

from graph_datasets.config import get_config as get_datasets_config

dataset = "synthetic"  # synthetic, msd, or disk
viz_lim = 3  # limit for visualizations (specifically for msd & disk)

if dataset == "synthetic":
    synteticdataset_settings = get_datasets_config("ifh")
    synteticdataset_settings["source"]["base_graphs"]["n_buildings"] = 3


    dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings, logger = None, report_path = "???", dataset_name = "test")
    dataset_generator.create_dataset()
    extended_nxdatset = dataset_generator.extend_nxdataset(dataset_generator.graphs["noise"], "training", "training")
    # extended_nxdatset = dataset_generator.extend_nxdataset(dataset_generator.graphs["original"], "training", "training")

    all_dataset = extended_nxdatset["train"] + extended_nxdatset["test"] +extended_nxdatset["val"]

elif dataset == "msd":
    synteticdataset_settings = get_datasets_config("msd")
    dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings, logger = None, report_path = "???", dataset_name = "test")
    # dataset_generator.create_dataset()
    visualized_graphs = dataset_generator.graphs["original"][:viz_lim]  # limit to viz_lim visualizations, while keeping the graphs pool.
    extended_nxdatset = dataset_generator.extend_nxdataset(visualized_graphs, "training", "training")
    # extended_nxdatset = dataset_generator.extend_nxdataset(dataset_generator.graphs["original"], "training", "training")

    all_dataset = extended_nxdatset["train"] + extended_nxdatset["test"] +extended_nxdatset["val"]
    print(f"dbg len(all_dataset) {len(all_dataset)}")

elif dataset == "disk":
    synteticdataset_settings = get_datasets_config("disk")
    dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings, logger = None, report_path = "???", dataset_name = "test")

    visualized_graphs = dataset_generator.graphs["original"][:viz_lim]  # same as msd, only visualize subset of full data pool
    extended_nxdatset = dataset_generator.extend_nxdataset(visualized_graphs, "training", "final")

    all_dataset = extended_nxdatset["train"] + extended_nxdatset["test"] +extended_nxdatset["val"]
    print(f"dbg len(all_dataset) {len(all_dataset)}")

    if len(all_dataset) == 0:
        print("all_datasets is empty, visualizing input graph")
        all_dataset = visualized_graphs

for graph in all_dataset:
    # node_types = graph.get_all_node_types()
    # print(f"dbg type(graph) {type(graph)}")
    # print(f'dbg node_types {node_types}')
    # if "door" in node_types:
    #     print(f'dbg door in node types!')
    if type(graph) == GraphWrapper:
        visualize_nxgraph_3d(graph, "train data", visualize_alone=True, include_node_ids=False)
        plt.show()
    elif type(graph) == list:
        print(f'dbg new sequence of graphs of length {len(graph)}')
        for i, graph_i in enumerate(graph):
            if i == len(graph) - 1:
                blocking = True
            else:
                blocking = False
            print(blocking)

            if type(graph_i) == GraphWrapper:
                visualize_nxgraph_3d(graph_i, f"train data {i}", visualize_alone=True, include_node_ids=False, blocking=blocking)
                # plt.show()