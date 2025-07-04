from SyntheticDatasetGenerator import SyntheticDatasetGenerator
from graph_visualizer import visualize_nxgraph, visualize_nxgraph_3d
import matplotlib.pyplot as plt
import json, os, time, sys
import networkx as nx


from graph_datasets.config import get_config as get_datasets_config
# from graph_reasoning.config import get_config as get_reasoning_config
synteticdataset_settings = get_datasets_config("msd")

dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings, logger = None, report_path = "???", dataset_name = "test")

# filtered_nxdataset = dataset_generator.get_filtered_datset(settings_hdata["nodes"],settings_hdata["edges"])["noise"]
extended_nxdatset = dataset_generator.extend_nxdataset(dataset_generator.graphs["original"], "training", "training")
# normalized_nxdatset = dataset_generator.normalize_features_nxdatset(extended_nxdatset)
# view1 = dataset_generator.graphs["views"][0].filter_graph_by_node_attributes_containted({"view" : 1})
# view2 = dataset_generator.graphs["views"][0].filter_graph_by_node_attributes_containted({"view" : 2})
# view3 = dataset_generator.graphs["views"][0].filter_graph_by_node_attributes_containted({"view" : 3})
# visualize_nxgraph(dataset_generator.graphs["original"][0], "original")
# visualize_nxgraph(dataset_generator.graphs["noise"][0], "noise")
# visualize_nxgraph(view1, "with views 1")
# visualize_nxgraph(view2, "with views 2")
# visualize_nxgraph(view3, "with views 3")

all_dataset = extended_nxdatset["train"] + extended_nxdatset["test"] +extended_nxdatset["val"]


for graph in dataset_generator.graphs["original"][:10]:
    # graph.remove_all_edges()
    # visualize_nxgraph_3d(graph, "train data", visualize_alone=True)
    # plt.show()

    centers_2d = centers = nx.get_node_attributes(graph.graph, 'center')
    labels = nx.get_node_attributes(graph.graph, 'type')
    pos = nx.spring_layout(graph.graph)
    edge_weights = nx.get_edge_attributes(graph.graph, 'weight')
    nx.draw(graph.graph, with_labels=True, labels=labels, node_color='lightblue', node_size=800, font_size=14)
    nx.draw_networkx_edge_labels(graph.graph, centers_2d, edge_labels=edge_weights)
    plt.title("Graph Visualization")
    plt.show()