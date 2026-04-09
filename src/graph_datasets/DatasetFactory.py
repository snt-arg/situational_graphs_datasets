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


class DatasetFactory:
    def __init__(self, 
                 dataset_base="synthetic", 
                 config_name="matching/incremental",
                 extension_name="original",
                 pickle_name="matching/incremental.pkl",
                 n_graphs_reduction=5,
                 save_pickle_path="/home/adminpc/workspaces/reasoning_ws/src/situational_graphs_datasets/datasets/",
                 logger=None):
        """
        Initialize DatasetFactory.
        
        Args:
            dataset_base: "synthetic", "msd", or "custom_pickle"
            config_name: name of the config file in graph_datasets/config
            extension_name: "original" or "noise"
            pickle_name: name for saving pickle file
            n_graphs_reduction: number of graphs to generate (None for all)
            save_pickle_path: path to save pickle files
            logger: logger instance
        """
        self.dataset_base = dataset_base
        self.config_name = config_name
        self.extension_name = extension_name
        self.pickle_name = pickle_name
        self.n_graphs_reduction = n_graphs_reduction
        self.save_pickle_path = save_pickle_path
        self.logger = logger
        self.full_save_path = os.path.join(save_pickle_path, pickle_name)
        
        self.dataset_generator = None
        self.all_dataset = None
        self.matching_dataset = None

    def create_synthetic_dataset(self):
        """Create synthetic dataset."""
        synteticdataset_settings = get_datasets_config(self.config_name)
        if self.n_graphs_reduction is not None:
            synteticdataset_settings["source"]["base_graphs"]["n_buildings"] = self.n_graphs_reduction

        self.dataset_generator = SyntheticDatasetGenerator(
            synteticdataset_settings, 
            logger=self.logger, 
            report_path="???", 
            dataset_name="test"
        )
        self.dataset_generator.create_dataset()
        extended_nxdatset = self.dataset_generator.extend_nxdataset(
            self.dataset_generator.graphs[self.extension_name], 
            "training", 
            "training"
        )

        self.all_dataset = extended_nxdatset["train"] + extended_nxdatset["test"] + extended_nxdatset["val"]
        
        return self.all_dataset

    def create_msd_dataset(self):
        """Create MSD dataset."""
        synteticdataset_settings = get_datasets_config(self.config_name)
        if self.n_graphs_reduction is not None:
            synteticdataset_settings["source"]["limit"] = self.n_graphs_reduction
        
        self.dataset_generator = SyntheticDatasetGenerator(
            synteticdataset_settings, 
            logger=self.logger, 
            report_path="???", 
            dataset_name="test"
        )
        extended_nxdatset = self.dataset_generator.extend_nxdataset(
            self.dataset_generator.graphs[self.extension_name], 
            "training", 
            "training"
        )

        self.all_dataset = extended_nxdatset["train"] + extended_nxdatset["test"] + extended_nxdatset["val"]
        self.all_dataset = [g[0] for g in self.all_dataset]
        
        return self.all_dataset

    def create_custom_pickle_dataset(self, pickle_path='/home/adminpc/dockers/s_graphs_jazzy/workspace/plane_graphs/multi_building/test/0/planes_graphs/merge_graph.joblib'):
        """Create dataset from custom pickle."""
        synteticdataset_settings = get_datasets_config(self.config_name)
        self.dataset_generator = SyntheticDatasetGenerator(
            synteticdataset_settings, 
            logger=self.logger, 
            report_path="???", 
            dataset_name="test"
        )
        self.all_dataset = [joblib.load(pickle_path)]
        
        return self.all_dataset

    def create_dataset(self):
        """Create dataset based on dataset_base type."""
        if self.dataset_base == "synthetic":
            return self.create_synthetic_dataset()
        elif self.dataset_base == "msd":
            return self.create_msd_dataset()
        elif self.dataset_base == "custom_pickle":
            return self.create_custom_pickle_dataset()
        else:
            raise ValueError(f"Unknown dataset_base: {self.dataset_base}")

    def visualize_dataset(self, dataset=None):
        """Visualize the dataset."""
        if dataset is None:
            dataset = self.all_dataset
            
        if dataset is None:
            print("No dataset to visualize. Create dataset first.")
            return
            
        for graph in dataset:
            if type(graph) == GraphWrapper:
                visualize_nxgraph_3d(graph, "train data", visualize_alone=True, include_node_ids=False)
                plt.show()
            elif type(graph) == list:
                for i, graph_i in enumerate(graph):
                    if i == len(graph) - 1:
                        blocking = True
                    else:
                        blocking = False

                    if type(graph_i) == GraphWrapper:
                        visualize_nxgraph_3d(
                            graph_i, 
                            f"train data {i}", 
                            visualize_alone=True, 
                            include_node_ids=False, 
                            blocking=blocking, 
                            show_hover_tooltips=True
                        )

    def save_dataset(self, dataset=None, save_path=None):
        """Save dataset to pickle."""
        if dataset is None:
            dataset = self.all_dataset
        if save_path is None:
            save_path = self.full_save_path
            
        if dataset is None:
            print("No dataset to save. Create dataset first.")
            return
            
        if self.dataset_generator is None:
            print("No dataset generator available for saving.")
            return
            
        self.dataset_generator.save_networkx_graphs_to_pickle(dataset, save_path, save_wrapper=True)
        print(f'Saved {self.config_name} dataset to {save_path}')

    def matching_extension(self):
        self.all_dataset = self.dataset_generator.compose_a_s_graphs(self.all_dataset)


    def run(self, visualize=False, save_pickle=False):
        """Run the complete dataset creation pipeline."""
        # Create dataset
        self.create_dataset()
        
        # Visualize if requested
        if visualize:
            self.visualize_dataset()
            
        # Save if requested
        if save_pickle:
            self.save_dataset()
            
        return self.all_dataset


# # Example usage for backward compatibility
# if __name__ == "__main__":
#     # Default configuration
#     factory = DatasetFactory(
#         dataset_base="msd",
#         config_name="kim/msd_buildings_objects",
#         extension_name="original",
#         pickle_name="kim/msd_buildings_objects.pkl",
#         n_graphs_reduction=50
#     )
    
#     all_dataset = factory.run(visualize=False, save_pickle=True)


