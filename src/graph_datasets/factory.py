from SyntheticDatasetGenerator import SyntheticDatasetGenerator
from graph_visualizer import visualize_nxgraph, visualize_nxgraph_3d
import matplotlib.pyplot as plt
import json, os, time, sys
import networkx as nx
from graph_wrapper.GraphWrapper import GraphWrapper
import pickle
import numpy as np
import joblib

from graph_datasets.config import get_config as get_datasets_config

dataset_base = "synthetic"  # "synthetic" or "msd" or "custom_pickle"
config_name = "ifh/ssg_manh_small_buildings"  # name of the config file in graph_datasets/config
extension_name = "original"  # "original" or "noise"
pickle_name = "ifh/viz.pkl"
n_graphs_reduction = 5
save_pickle = False
visualize = True

save_pickle_path = "/home/adminpc/workspaces/reasoning_ws/src/situational_graphs_datasets/datasets/"
full_save_path = os.path.join(save_pickle_path, pickle_name)

if dataset_base == "synthetic":
    synteticdataset_settings = get_datasets_config(config_name)
    if n_graphs_reduction is not None:
        synteticdataset_settings["source"]["base_graphs"]["n_buildings"] = n_graphs_reduction


    dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings, logger = None, report_path = "???", dataset_name = "test")
    dataset_generator.create_dataset()
    extended_nxdatset = dataset_generator.extend_nxdataset(dataset_generator.graphs[extension_name], "training", "training")
    # extended_nxdatset = dataset_generator.extend_nxdataset(dataset_generator.graphs["original"], "training", "training")

    all_dataset = extended_nxdatset["train"] + extended_nxdatset["test"] +extended_nxdatset["val"]
    all_dataset = [g[0] for g in all_dataset]

elif dataset_base == "msd":
    synteticdataset_settings = get_datasets_config(config_name)
    if n_graphs_reduction is not None:
        synteticdataset_settings["source"]["limit"] = n_graphs_reduction
    
    dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings, logger = None, report_path = "???", dataset_name = "test")
    # dataset_generator.create_dataset()
    extended_nxdatset = dataset_generator.extend_nxdataset(dataset_generator.graphs[extension_name], "training", "training")
    # extended_nxdatset = dataset_generator.extend_nxdataset(dataset_generator.graphs["original"], "training", "training")

    all_dataset = extended_nxdatset["train"] + extended_nxdatset["test"] +extended_nxdatset["val"]
    all_dataset = [g[0] for g in all_dataset]

elif dataset_base == "custom_pickle":
    synteticdataset_settings = get_datasets_config(config_name)
    dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings, logger = None, report_path = "???", dataset_name = "test")
    all_dataset = [joblib.load('/home/adminpc/dockers/s_graphs_jazzy/workspace/plane_graphs/multi_building/test/0/planes_graphs/merge_graph.joblib')]

if visualize:
    for graph in all_dataset:
        if type(graph) == GraphWrapper:
            visualize_nxgraph_3d(graph, "train data", visualize_alone=True, include_node_ids=False)
            plt.show()
        elif type(graph) == list:
            # print(f'dbg new sequence of graphs of length {len(graph)}')
            for i, graph_i in enumerate(graph):
                if i == len(graph) - 1:
                    blocking = True
                else:
                    blocking = False

                if type(graph_i) == GraphWrapper:
                    visualize_nxgraph_3d(graph_i, f"train data {i}", visualize_alone=True, include_node_ids=False, blocking=blocking)
                    # plt.show()

if save_pickle:
    dataset_generator.save_networkx_graphs_to_pickle(all_dataset, full_save_path)
    print(f'Saved {config_name} dataset to {full_save_path}')



# for key in ["ws", "wall", "room", "floor"]:
#     graph = all_dataset[0].filter_graph_by_node_types([key])
    
#     center = list(graph.get_attributes_of_all_nodes())[0][1]['normal']
#     print(f'key {key}, type of center attribute: {type(center)}, element type {type(center[0])}, size {len(center)}, value {center}')