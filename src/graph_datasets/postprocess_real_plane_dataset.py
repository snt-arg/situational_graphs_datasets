import matplotlib
matplotlib.use("TkAgg")   # or "Qt5Agg"
import matplotlib.pyplot as plt
import os
import sys
import csv
import json
import time
import random
import argparse
import traceback
from typing import Any, Dict, List, Tuple
import networkx as nx
import copy
import ot

import tqdm

import plotly.graph_objects as go
import numpy as np
import torch

# ===== lightweight logger =====
class SimpleLogger:
    def __init__(self, name="metrics"):
        self.name = name
    def info(self, msg):  print(f"[INFO] [{self.name}] {msg}")
    def warn(self, msg):  print(f"[WARN] [{self.name}] {msg}")
    def error(self, msg): print(f"[ERROR] [{self.name}] {msg}")

# ===== repo imports (pure-Python) =====
from graph_wrapper.GraphWrapper import GraphWrapper
from graph_datasets.SyntheticDatasetGenerator import SyntheticDatasetGenerator
from graph_datasets.graph_visualizer import visualize_nxgraph_3d, visualize_nxgraph
import graph_datasets.graph_visualizer as gv
print("graph_visualizer loaded from:", gv.__file__)

from graph_datasets.config import get_config as datasets_get_config
from graph_reasoning.GNNWrapper import GNNWrapper as GNNWrapper2
from graph_reasoning.config import get_config as reasoning_get_config
from graph_reasoning.pths import get_pth as reasoning_get_pth
from graph_factor_nn.FactorNNBridge import FactorNNBridge
from graph_factor_nn.FactorNN import FactorNN
from graph_matching.utils import segments_distance, segment_intersection, plane_6_params_to_4_params



class PostprocessRealPlaneDataset:
    def __init__(self, dataset_tag, output_path):
        print("Instantiating PostprocessRealPlaneDataset...")
        self.dataset_tag = dataset_tag
        self.output_path = output_path

        self.postprocess_real_plane_dataset()

    def postprocess_real_plane_dataset(self, visualize=False, save_pickle=True):
        ds_settings = datasets_get_config(self.dataset_tag)

        ds_settings["training_split"]["val"] = 0.0
        ds_settings["training_split"]["test"] = 0.0
        
        self.dataset_generator = SyntheticDatasetGenerator(
            ds_settings,
            logger=None,
            report_path="???",
            dataset_name="test",
        )

        for graph in self.dataset_generator.graphs["original"]:
            for node in graph.get_attributes_of_all_nodes():
                if node[1]["type"] == "ws":
                    distance = np.linalg.norm(node[1]["limits"][1][0] - node[1]["limits"][0][0])
                    node[1]["length"] = distance


        if visualize:
            for i, graph in enumerate(self.dataset_generator.graphs["original"]):
                visualize_nxgraph_3d(graph, f"Initial Graph {i}", visualize_alone=True, include_node_ids=False, add_legend=True)

        self.dataset_generator.define_norm_limits()
        extended_nxdatset = self.dataset_generator.extend_nxdataset(self.dataset_generator.graphs["original"], "training", "preprocess_real")
        all_dataset = extended_nxdatset["train"] + extended_nxdatset["test"] +extended_nxdatset["val"]
        all_dataset = [g[0] for g in all_dataset]
        for i, graph in enumerate(all_dataset):
            print(f'dbg all edge types {graph.get_all_edge_types()}')
        if visualize:
            for i, graph in enumerate(all_dataset):
                
                visualize_nxgraph_3d(graph, f"Postprocessed Graph {i}", visualize_alone=True, include_node_ids=False, add_legend=True)
            plt.show()

        if save_pickle:
            self.dataset_generator.save_networkx_graphs_to_pickle(all_dataset, self.output_path)
            print(f'Saved dataset to {self.output_path}')



postprocesor = PostprocessRealPlaneDataset("ifh/preprocess_real", "/home/adminpc/workspaces/reasoning_ws/src/situational_graphs_datasets/datasets/ifh/real/ssg_real_buildings.pkl")