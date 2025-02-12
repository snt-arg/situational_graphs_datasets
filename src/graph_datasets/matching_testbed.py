from SyntheticDatasetGenerator import SyntheticDatasetGenerator
import graph_visualizer as gv
import json, os, time, sys
import networkx as nx
from graph_datasets.config import get_config as get_datasets_config

# from graph_reasoning.config import get_config as get_reasoning_config
synteticdataset_settings = get_datasets_config("graph_matching")
dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings, logger = None, report_path = "AS_Datasets", dataset_name = "test")
dataset_generator.create_dataset()
a_graphs_list = dataset_generator.graphs["original"]
s_graphs_list = dataset_generator.graphs["noise"]
extended_list = dataset_generator.graphs["extended"]
# filtered_nxdataset = dataset_generator.get_filtered_datset(settings_hdata["nodes"],settings_hdata["edges"])["noise"]

extended_nxdatset = dataset_generator.extend_nxdataset(dataset_generator.graphs["noise"], "training", "training")

# normalized_nxdatset = dataset_generator.normalize_features_nxdatset(extended_nxdatset)
# view1 = dataset_generator.graphs["views"][0].filter_graph_by_node_attributes_containted({"view" : 1})
# view2 = dataset_generator.graphs["views"][0].filter_graph_by_node_attributes_containted({"view" : 2})
# view3 = dataset_generator.graphs["views"][0].filter_graph_by_node_attributes_containted({"view" : 3})
# visualize_nxgraph(dataset_generator.graphs["original"][0], "original")
# visualize_nxgraph(dataset_generator.graphs["noise"][0], "noise")
# visualize_nxgraph(view1, "with views 1")
# visualize_nxgraph(view2, "with views 2")
# visualize_nxgraph(view3, "with views 3")

# all_dataset = extended_nxdatset["train"] + extended_nxdatset["test"] +extended_nxdatset["val"]

# serialize dataset
dataset_generator.serialize_dataset()
# for i in range(len(s_graphs_list)):
#     gv.visualize_nxgraph(a_graphs_list[i], "original", visualize_alone=True)
#     gv.visualize_nxgraph(s_graphs_list[i], "noise", visualize_alone=True)
#     gv.visualize_nxgraph(dataset_generator.graphs["extended"][i], "extended", visualize_alone=True)


# print(len(a_graphs_list))
# print(len(s_graphs_list))
# print(len(extended_list))

# deserialize dataset
# dataset_generator.deserialize_dataset()
# a_graphs_list = dataset_generator.graphs["original"]
# s_graphs_list = dataset_generator.graphs["noise"]
# extended_list = dataset_generator.graphs["extended"]

# print(len(a_graphs_list))
# print(len(s_graphs_list))
# print(len(extended_list))

# for i in range(len(s_graphs_list)):
#     gv.visualize_nxgraph_pair(a_graphs_list[i], s_graphs_list[i], f"orginal_noise{i}", visualize_alone=True)
#     gv.visualize_nxgraph(dataset_generator.graphs["extended"][i], f"extended{i}", visualize_alone=True)

# # convert graphwrapper to digraph
# imported = nx.DiGraph()
# imported.add_nodes_from(a_graphs_list[0].get_attributes_of_all_nodes())
# imported.add_edges_from(a_graphs_list[0].get_attributes_of_all_edges())
# gv.visualize_digraph(imported, "imported")
# gv.visualize_nxgraph_pair(a_graphs_list[0], imported, "imported", visualize_alone=True, g1digraph=False, g2digraph=True)