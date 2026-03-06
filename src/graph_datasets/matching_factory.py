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

from graph_datasets.DatasetFactory import DatasetFactory

# Default configuration
factory = DatasetFactory(
    dataset_base="synthetic",
    config_name="matching/incremental_symmetries",
    extension_name="original",
    pickle_name="matching/incremental_symmetries_grid_squared_translation.pkl",
    n_graphs_reduction=5
)

visualize = True
save_pickle = True

# Run with visualization and saving
"""Run the complete dataset creation pipeline."""
# Create dataset
factory.create_dataset()
factory.matching_extension()

# Visualize if requested
for (a_graph, s_graph_list) in factory.all_dataset:
    
    if type(a_graph) == GraphWrapper and visualize:
        visualize_nxgraph_3d(a_graph, "Agraph", visualize_alone=True, include_node_ids=False, blocking=True, show_hover_tooltips=True, hide_axes=False)
    
    # for i, graph in enumerate(s_graph_list):
    #     if type(graph) == GraphWrapper and visualize:
    #         visualize_nxgraph_3d(graph, f"Sgraph {i}", visualize_alone=True, include_node_ids=False, blocking=False, show_hover_tooltips=True, hide_axes=False)

    # input("asdf")
    
# Save if requested
if save_pickle:
    factory.save_dataset()


