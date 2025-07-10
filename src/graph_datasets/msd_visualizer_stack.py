from SyntheticDatasetGenerator import SyntheticDatasetGenerator
from graph_visualizer import visualize_nxgraph, visualize_nxgraph_3d
import matplotlib.pyplot as plt
import json, os, time, sys
import networkx as nx


from graph_datasets.config import get_config as get_datasets_config
# from graph_reasoning.config import get_config as get_reasoning_config
synteticdataset_settings = get_datasets_config("ifh")

synteticdataset_settings["source"]["base_graphs"]["n_buildings"] = 10
dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings, logger = None, report_path = "???", dataset_name = "test")
dataset_generator.create_dataset()
# filtered_nxdataset = dataset_generator.get_filtered_datset(settings_hdata["nodes"],settings_hdata["edges"])["noise"]
extended_nxdatset = dataset_generator.extend_nxdataset(dataset_generator.graphs["original"], "training", "training")

all_dataset = extended_nxdatset["test"] + extended_nxdatset["test"] +extended_nxdatset["val"]


for graph in all_dataset[:10]:
    visualize_nxgraph_3d(graph, "train data", visualize_alone=True, include_node_ids=False)
    plt.show()

    # centers_2d = centers = nx.get_node_attributes(graph.graph, 'center')
    # labels = nx.get_node_attributes(graph.graph, 'type')
    # pos = nx.spring_layout(graph.graph)
    # edge_weights = nx.get_edge_attributes(graph.graph, 'weight')
    # nx.draw(graph.graph, with_labels=True, labels=labels, node_color='lightblue', node_size=800, font_size=14)
    # nx.draw_networkx_edge_labels(graph.graph, centers_2d, edge_labels=edge_weights)
    # plt.title("Graph Visualization")
    # plt.show()