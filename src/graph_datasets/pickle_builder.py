from SyntheticDatasetGenerator import SyntheticDatasetGenerator
from graph_visualizer import visualize_nxgraph, visualize_nxgraph_3d
from InteractiveGraphVisualizer import InteractiveGraphVisualizer

import matplotlib.pyplot as plt
import json, os, time, sys


from graph_datasets.config import get_config as get_datasets_config
from graph_reasoning.config import get_config as get_reasoning_config

dataset = "synthetic"

if dataset == "synthetic":
    synteticdataset_settings = get_datasets_config("ifh")
    synteticdataset_settings["source"]["base_graphs"]["n_buildings"] = 3000


    dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings, logger = None, report_path = "???", dataset_name = "test")
    dataset_generator.create_dataset()
    extended_nxdatset = dataset_generator.extend_nxdataset(dataset_generator.graphs["noise"], "training", "training")
    # extended_nxdatset = dataset_generator.extend_nxdataset(dataset_generator.graphs["original"], "training", "training")

    all_dataset = extended_nxdatset["train"] + extended_nxdatset["test"] +extended_nxdatset["val"]


    for graph in all_dataset:
        print(f"dbg type(graph) {type(graph)}")
    #     visualize_nxgraph_3d(graph, "train data", visualize_alone=True, include_node_ids=False)
    #     plt.show()1

    save_path = "/home/adminpc/workspaces/reasoning_ws/src/situational_graphs_datasets/datasets/test/ssg_Lshaped_noise_3000.pkl"
    # dataset_generator.save_wrappers_to_pickle(extended_nxdatset['train'], save_path)
    dataset_generator.save_networkx_graphs_to_pickle(all_dataset, save_path)


elif dataset == "msd":
    synteticdataset_settings = get_datasets_config("msd")
    dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings, logger = None, report_path = "???", dataset_name = "test")
    # dataset_generator.create_dataset()
    extended_nxdatset = dataset_generator.extend_nxdataset(dataset_generator.graphs["original"], "training", "training")
    # extended_nxdatset = dataset_generator.extend_nxdataset(dataset_generator.graphs["original"], "training", "training")

    all_dataset = extended_nxdatset["train"] + extended_nxdatset["test"] +extended_nxdatset["val"]
    print(f"dbg len(all_dataset) {len(all_dataset)}")

# for graph in all_dataset:
#     # node_types = graph.get_all_node_types()
#     # print(f"dbg type(graph) {type(graph)}")
#     # print(f'dbg node_types {node_types}')
#     # if "door" in node_types:
#     #     print(f'dbg door in node types!')
#     visualize_nxgraph_3d(graph, "train data", visualize_alone=True, include_node_ids=False)
#     plt.show()

save_path = "/home/adminpc/workspaces/reasoning_ws/src/situational_graphs_datasets/datasets/test/ssg_Lshaped_big_3000.pkl"
# dataset_generator.save_wrappers_to_pickle(extended_nxdatset['train'], save_path)
dataset_generator.save_networkx_graphs_to_pickle(all_dataset, save_path)