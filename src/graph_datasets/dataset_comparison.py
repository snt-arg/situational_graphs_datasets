from SyntheticDatasetGenerator import SyntheticDatasetGenerator
from graph_visualizer import visualize_nxgraph, visualize_nxgraph_3d
import matplotlib.pyplot as plt
import json, os, time, sys
import networkx as nx
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import numpy as np
import pickle

from graph_datasets.config import get_config as get_datasets_config

### SSG
filename = "/home/adminpc/workspaces/reasoning_ws/src/situational_graphs_datasets/datasets/test/ssg_manhattan_small_20.pkl"
if os.path.exists(filename):
    with open(filename, 'rb') as f:
        ssg_all_dataset = pickle.load(f)

else:

    synteticdataset_settings = get_datasets_config("ifh")
    synteticdataset_settings["source"]["base_graphs"]["n_buildings"] = 100


    dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings, logger = None, report_path = "???", dataset_name = "test")
    dataset_generator.create_dataset()
    extended_nxdatset = dataset_generator.extend_nxdataset(dataset_generator.graphs["noise"], "training", "training")
    # extended_nxdatset = dataset_generator.extend_nxdataset(dataset_generator.graphs["original"], "training", "training")

    ssg_all_dataset = extended_nxdatset["train"] + extended_nxdatset["test"] +extended_nxdatset["val"]


### SSG
filename = "/home/adminpc/workspaces/reasoning_ws/src/situational_graphs_datasets/datasets/test/msd_100.pkl"
if os.path.exists(filename):
    with open(filename, 'rb') as f:
        msd_all_dataset = pickle.load(f)

else:
    synteticdataset_settings = get_datasets_config("msd")
    dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings, logger = None, report_path = "???", dataset_name = "test")
    # dataset_generator.create_dataset()
    extended_nxdatset = dataset_generator.extend_nxdataset(dataset_generator.graphs["original"], "training", "training")
    # extended_nxdatset = dataset_generator.extend_nxdataset(dataset_generator.graphs["original"], "training", "training")

    msd_all_dataset = extended_nxdatset["train"] + extended_nxdatset["test"] +extended_nxdatset["val"]


synteticdataset_settings = get_datasets_config("ifh")
dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings, logger = None, report_path = "???", dataset_name = "test")

def plot_comparison(stats1, stats2, label1="List 1", label2="List 2"):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # ---- Nodes (avg as bar, max as line) ----
    axs[0, 0].bar([label1, label2], [stats1["avg_nodes"], stats2["avg_nodes"]], label='Average')
    axs[0, 0].axhline(stats1["max_nodes"], color='blue', linestyle='--', label=f'{label1} Max')
    axs[0, 0].axhline(stats2["max_nodes"], color='orange', linestyle='--', label=f'{label2} Max')
    axs[0, 0].set_title("Number of Nodes")
    axs[0, 0].legend()

    # ---- Edges (avg as bar, max as line) ----
    axs[0, 1].bar([label1, label2], [stats1["avg_edges"], stats2["avg_edges"]], label='Average')
    axs[0, 1].axhline(stats1["max_edges"], color='blue', linestyle='--', label=f'{label1} Max')
    axs[0, 1].axhline(stats2["max_edges"], color='orange', linestyle='--', label=f'{label2} Max')
    axs[0, 1].set_title("Number of Edges")
    axs[0, 1].legend()

    # ---- Nodes per type (avg as bar, max as line) ----
    all_types = sorted(set(stats1["avg_node_types"].keys()) | set(stats2["avg_node_types"].keys()))
    x = np.arange(len(all_types))
    width = 0.35

    avg1 = [stats1["avg_node_types"].get(t, 0) for t in all_types]
    avg2 = [stats2["avg_node_types"].get(t, 0) for t in all_types]
    max1 = [stats1["max_node_types"].get(t, 0) for t in all_types]
    max2 = [stats2["max_node_types"].get(t, 0) for t in all_types]

    axs[1, 0].bar(x - width/2, avg1, width, label=f'{label1} Avg')
    axs[1, 0].bar(x + width/2, avg2, width, label=f'{label2} Avg')

    axs[1, 0].plot(x - width/2, max1, 'o--', color='blue', label=f'{label1} Max')
    axs[1, 0].plot(x + width/2, max2, 'o--', color='orange', label=f'{label2} Max')

    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(all_types, rotation=45)
    axs[1, 0].set_title("Nodes per Type")
    axs[1, 0].legend()

    # ---- Degree Distribution ----
    degs1 = stats1["degree_distribution"]
    degs2 = stats2["degree_distribution"]
    degs1_hist = Counter(degs1)
    degs2_hist = Counter(degs2)
    all_degrees = sorted(set(degs1_hist.keys()) | set(degs2_hist.keys()))
    d1 = [degs1_hist.get(d, 0) / len(degs1) for d in all_degrees]
    d2 = [degs2_hist.get(d, 0) / len(degs2) for d in all_degrees]

    axs[1, 1].plot(all_degrees, d1, marker='o', label=label1)
    axs[1, 1].plot(all_degrees, d2, marker='x', label=label2)
    axs[1, 1].set_title("Degree Distribution")
    axs[1, 1].set_xlabel("Degree")
    axs[1, 1].set_ylabel("Fraction of Nodes")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    # Degree distribution (separate figure)
    degs1 = stats1["degree_distribution"]
    degs2 = stats2["degree_distribution"]
    degs1_hist = Counter(degs1)
    degs2_hist = Counter(degs2)
    all_degrees = sorted(set(degs1_hist.keys()) | set(degs2_hist.keys()))
    d1 = [degs1_hist.get(d, 0) / len(degs1) for d in all_degrees]
    d2 = [degs2_hist.get(d, 0) / len(degs2) for d in all_degrees]

    plt.figure(figsize=(8, 5))
    plt.plot(all_degrees, d1, marker='o', label=label1)
    plt.plot(all_degrees, d2, marker='x', label=label2)
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Fraction of Nodes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

stats1 = dataset_generator.compute_stats(ssg_all_dataset)
stats2 = dataset_generator.compute_stats(msd_all_dataset)
plot_comparison(stats1, stats2, label1="Semantic", label2="Real")