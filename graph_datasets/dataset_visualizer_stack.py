from SyntheticDatasetGenerator import SyntheticDatasetGenerator
from graph_visualizer import visualize_nxgraph
import matplotlib.pyplot as plt
import json, os, time, sys

graph_wrapper_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_reasoning","graph_reasoning")
sys.path.append(graph_wrapper_dir)
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"config", "dataset_testing.json")) as f:
    synteticdataset_settings = json.load(f)
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"situational_graphs_reasoning" , "config", "pard_training.json")) as f:
    graph_reasoning_settings = json.load(f)

dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings, logger = None, report_path = None, dataset_name = "test")
dataset_generator.create_dataset()
settings_hdata = graph_reasoning_settings["hdata"]
filtered_nxdataset = dataset_generator.get_filtered_datset(settings_hdata["nodes"],settings_hdata["edges"])["noise"]
# extended_nxdatset = dataset_generator.extend_nxdataset(filtered_nxdataset, settings_hdata["edges"][0][1], "training")
# normalized_nxdatset = dataset_generator.normalize_features_nxdatset(extended_nxdatset)
merged_graph, slices = dataset_generator.merge_graphs_type_as_x(filtered_nxdataset)
print(f"dbg merged_graph x {merged_graph.x}")
print(f"dbg merged_graph edge_attrs {merged_graph.edge_attr}")
print(f"dbg slices {slices}")
# view1 = dataset_generator.graphs["views"][0].filter_graph_by_node_attributes_containted({"view" : 1})
# view2 = dataset_generator.graphs["views"][0].filter_graph_by_node_attributes_containted({"view" : 2})
# view3 = dataset_generator.graphs["views"][0].filter_graph_by_node_attributes_containted({"view" : 3})
# visualize_nxgraph(dataset_generator.graphs["original"][0], "original")
# visualize_nxgraph(dataset_generator.graphs["noise"][0], "noise")
# visualize_nxgraph(view1, "with views 1")
# visualize_nxgraph(view2, "with views 2")
# visualize_nxgraph(view3, "with views 3")

# visualize_nxgraph(filtered_nxdataset[0], "train data")
# plt.show()