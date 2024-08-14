from SyntheticDatasetGenerator import SyntheticDatasetGenerator
from graph_visualizer import visualize_nxgraph
import matplotlib.pyplot as plt
import json, os, time, sys


from graph_datasets.config import get_config as get_datasets_config
from graph_reasoning.config import get_config as get_reasoning_config
synteticdataset_settings = get_datasets_config("dataset_testing")
graph_reasoning_settings = get_reasoning_config(f"room_merge")

dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings, logger = None, report_path = None, dataset_name = "test")
dataset_generator.create_dataset()
settings_hdata = graph_reasoning_settings["hdata"]
filtered_nxdataset = dataset_generator.get_filtered_datset(settings_hdata["nodes"],settings_hdata["edges"])["original"]
extended_nxdatset = dataset_generator.extend_nxdataset(filtered_nxdataset, "training", "training")
# normalized_nxdatset = dataset_generator.normalize_features_nxdatset(extended_nxdatset)
# view1 = dataset_generator.graphs["views"][0].filter_graph_by_node_attributes_containted({"view" : 1})
# view2 = dataset_generator.graphs["views"][0].filter_graph_by_node_attributes_containted({"view" : 2})
# view3 = dataset_generator.graphs["views"][0].filter_graph_by_node_attributes_containted({"view" : 3})
# visualize_nxgraph(dataset_generator.graphs["original"][0], "original")
# visualize_nxgraph(dataset_generator.graphs["noise"][0], "noise")
# visualize_nxgraph(view1, "with views 1")
# visualize_nxgraph(view2, "with views 2")
# visualize_nxgraph(view3, "with views 3")

visualize_nxgraph(filtered_nxdataset[0], "train data")
plt.show()