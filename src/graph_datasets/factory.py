from SyntheticDatasetGenerator import SyntheticDatasetGenerator
from graph_visualizer import visualize_nxgraph, visualize_nxgraph_3d
import matplotlib.pyplot as plt
import json, os, time, sys
import networkx as nx
from graph_wrapper.GraphWrapper import GraphWrapper


from graph_datasets.config import get_config as get_datasets_config

dataset_base = "synthetic"  # "synthetic" or "msd"
config_name = "ifh"
extension_name = "noise"
pickle_name = "ssg_manh_noise_small.pkl"
n_graphs_reduction = 3
save_pickle = False
visualize = True

save_pickle_path = "/home/adminpc/workspaces/reasoning_ws/src/situational_graphs_datasets/datasets/test/"
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

elif dataset_base == "msd":
    synteticdataset_settings = get_datasets_config(config_name)
    dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings, logger = None, report_path = "???", dataset_name = "test")
    # dataset_generator.create_dataset()
    extended_nxdatset = dataset_generator.extend_nxdataset(dataset_generator.graphs[extension_name], "training", "training")
    # extended_nxdatset = dataset_generator.extend_nxdataset(dataset_generator.graphs["original"], "training", "training")

    all_dataset = extended_nxdatset["train"] + extended_nxdatset["test"] +extended_nxdatset["val"]

if visualize:
    for graph in all_dataset:
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

if save_pickle:
    dataset_generator.save_networkx_graphs_to_pickle(all_dataset, full_save_path)