import numpy as np
import copy
import itertools
import random, math, time
import tqdm
from tqdm.contrib.concurrent import process_map
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import KDTree
from colorama import Fore, Back, Style
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch
from shapely.geometry import Polygon, Point
from collections import defaultdict, Counter
from typing import Optional

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import networkx as nx
import pickle

import sys
import os
import ast

import plot as pl

# graph_wrapper_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_wrapper")
# sys.path.append(graph_wrapper_dir)
from graph_wrapper.GraphWrapper import GraphWrapper
# graph_datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_datasets")
# sys.path.append(graph_datasets_dir)
from graph_datasets.graph_visualizer import visualize_nxgraph, visualize_nxgraph_3d
from graph_datasets.NodeEdgeFeatureEmbeddingBuildier import NodeEdgeFeatureEmbeddingBuildier
# graph_matching_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_matching")
# sys.path.append(graph_matching_dir)
from graph_matching.utils import relative_positions, segments_distance, closest_point_on_segment, distance_between_points, are_segments_collinear, relative_geometry
# graph_reasoning_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_reasoning")
# sys.path.append(graph_reasoning_dir)

viz_data_base = {"type": "Point", "feat": 'ro', "data": np.array([]), "linewidth": 1, "alpha": 1.0, "size": 1}

class SyntheticDatasetGenerator():

    def __init__(self, settings, logger = None, report_path = "", dataset_name = "", seed=None):
        print(f"SyntheticDatasetGenerator:", Fore.GREEN + "Initializing" + Fore.WHITE)
        self.settings = self.correct_json_initfeat_keys(settings)
        self.logger = logger
        self.report_path = report_path
        self.dataset_name = dataset_name

        if seed:
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
        self.seed = seed

        # dynamic save dir (either from settings or relative to file path)
        if "save_dir" in self.settings:
            self.save_dir = Path(self.settings["save_dir"])
        else:
            self.save_dir = Path(__file__).parent / "output_dataset"

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), self.report_path, self.dataset_name)
        self.graphs = {"original":[],"noise":[],"views":[],"extended":[]}
        
        self.define_viz_settings()

        if settings["source"]["type"] == "synthetic":
            self.max_n_rooms = 0
            self.define_norm_limits()

        elif settings["source"]["type"] == "msd":
            self.dataset_from_msd(settings["source"]["pickle_path"])  # expects file

        elif settings["source"]["type"] == "disk":
            self.dataset_from_disk(settings["source"]["folder_path"])  # expects folder
            

    def correct_json_initfeat_keys(self, settings):
        new_settings = copy.deepcopy(settings)
        for key, value in settings["initial_features"]["edges"].items():
            new_key = tuple([item.strip() for item in key.strip("[]").split(",")])
            new_settings["initial_features"]["edges"][new_key] = value
            new_settings["initial_features"]["edges"].pop(key)

        return new_settings

    def define_norm_limits(self):
        grid_dims = self.settings["source"]["base_graphs"]["grid_dims"]
        max_grid_dims = max(grid_dims[0][1],grid_dims[1][1])
        max_room_center_distances = self.settings["source"]["base_graphs"]["room_center_distances"][-1]
        max_room_entry_size = self.settings["source"]["base_graphs"]["max_room_entry_size"][-1]
        init_feat_keys = self.settings["initial_features"]
        max_building_size = max_grid_dims*max_room_center_distances
        max_room_size = max_room_entry_size*max_room_center_distances
        max_wall_thickness = grid_dims = self.settings["source"]["base_graphs"]["wall_thickness"][1]

        def add_features(type, feature_keys, working_dict):
            if type == "node":
                if feature_keys[0] == "centroid":
                    working_dict["min"] = np.concatenate([working_dict["min"], 0])
                    working_dict["max"] = np.concatenate([working_dict["max"], max_building_size])
                elif feature_keys[0] == "length":
                    working_dict["min"] = np.concatenate([working_dict["min"], [0]])
                    working_dict["max"] = np.concatenate([working_dict["max"], [max_room_size]]) #, [np.log(max_room_entry_size*max_room_center_distances)]])
                elif feature_keys[0] == "normals":
                    working_dict["min"] = np.concatenate([working_dict["min"],[-1,-1]])
                    working_dict["max"] = np.concatenate([working_dict["max"],[1,1]])

            elif type == "edge":
                if feature_keys[0] == "relative_pos":
                    working_dict["min"] = np.concatenate([working_dict["min"],-np.array([max_building_size,max_building_size])])
                    working_dict["max"] = np.concatenate([working_dict["max"],np.array([max_building_size,max_building_size])])
                elif feature_keys[0] == "min_dist":
                    working_dict["min"] = np.concatenate([working_dict["min"],[0]])
                    working_dict["max"] = np.concatenate([working_dict["max"],[max_building_size*np.sqrt(2)+max_wall_thickness*5]])  #,[np.log(max(playground_size)+1)]])
                elif feature_keys[0] == "centroids_distance":
                    working_dict["min"] = np.concatenate([working_dict["min"],[0]])
                    working_dict["max"] = np.concatenate([working_dict["max"],[max_building_size*np.sqrt(2)+max_wall_thickness*5]]) 
                elif feature_keys[0] == "angle_centroid_degrees":
                    working_dict["min"] = np.concatenate([working_dict["min"],[0]])
                    working_dict["max"] = np.concatenate([working_dict["max"],[360]]) 
                elif feature_keys[0] == "relative_ang_normal":
                    working_dict["min"] = np.concatenate([working_dict["min"],[0]])
                    working_dict["max"] = np.concatenate([working_dict["max"],[360]]) 

            if len(feature_keys) > 1:
                working_dict = add_features(type, feature_keys[1:], working_dict)
            return working_dict

        self.norm_limits = {"node" : add_features("node", init_feat_keys["nodes"]["ws"], {"min": [], "max":[]}), \
                            "edge" : add_features("edge", init_feat_keys["edges"][tuple(["ws","ws"])], {"min": [], "max":[]})}
        
    def define_viz_settings(self):
        self.node_viz_feat_mapping = {
            'ws': "black",
            'room': 'ro',
            'wall': 'oo',
            'floor': 'go',
            'building': 'co',
            'wall_ws': 'yo'
        }

        self.viz_center_offsets = {"ws": np.array([0, 0, 0]), "room": np.array([0, 0, 2]), "wall": np.array([0, 0, 1]),\
                                   "floor": np.array([0, 0, 3]), "building": np.array([0, 0, 2]), "object": np.array([0, 0, 0.5]), "city": np.array([0, 0, 4])}
                

    def normalize_features(self, type, feats):
        if len(feats) != 0:
            feats_norm = (feats-self.norm_limits[type]["min"])/(self.norm_limits[type]["max"]-self.norm_limits[type]["min"])
        else:
            feats_norm = []
        return feats_norm

    def create_dataset(self):
        print(f"SyntheticDatasetGenerator: ", Fore.GREEN + "Generating Syntetic Dataset" + Fore.WHITE)
        n_buildings = self.settings["source"]["base_graphs"]["n_buildings"]

        def process_building(_):
            base_matrix = self.generate_base_matrix()
            # original_graph = self.generate_graph_from_base_matrix(base_matrix, add_noise=False)
            original_graph = self.generate_graph_from_base_matrix(base_matrix, add_noise=False)
            noisy_graph = self.generate_graph_from_base_matrix(base_matrix, add_noise=True)
            return original_graph, noisy_graph

        # Using ThreadPoolExecutor for parallel processing
        num_workers = os.cpu_count()  # Or some fraction of it
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_building, i): i for i in range(n_buildings)}
            for future in tqdm.tqdm(as_completed(futures), total=n_buildings, colour="green"):
                original_graph, noisy_graph = future.result()
                self.graphs["original"].append(original_graph)
                self.graphs["noise"].append(noisy_graph)

        # for n_building in tqdm.tqdm(range(n_buildings), colour="green"):
        #     base_matrix = self.generate_base_matrix()

        #     self.graphs["original"].append(self.generate_graph_from_base_matrix(base_matrix, add_noise= False))
        #     self.graphs["noise"].append(self.generate_graph_from_base_matrix(base_matrix, add_noise= True))
        #     # self.graphs["views"].append(self.generate_graph_from_base_matrix(base_matrix, add_noise= False, add_multiview=True))
        # fig = plt.figure(constrained_layout=True)
        # fig.suptitle('Nodes histogram')
        # plt.show()
        # time.sleep(999)

    def generate_base_matrix(self):
        base_graph_settings = self.settings["source"]["base_graphs"]
        grid_dims = [np.random.randint(base_graph_settings["grid_dims"][0][0], base_graph_settings["grid_dims"][0][1] + 1),
                     np.random.randint(base_graph_settings["grid_dims"][1][0], base_graph_settings["grid_dims"][1][1] + 1)]
        max_room_entry_size = np.random.randint(base_graph_settings["max_room_entry_size"][0], base_graph_settings["max_room_entry_size"][1] + 1)
        min_room_entry_size = np.random.randint(base_graph_settings["min_room_entry_size"][0], base_graph_settings["min_room_entry_size"][1] + 1)
        
        # Check for room symmetries configuration
        room_similar_dimensions = base_graph_settings.get("symmetries", {}).get("room_similar_dimensions", 0)
        
        # Generate shared dimensions if symmetries are enabled
        shared_dim_x = None
        shared_dim_y = None
        
        if room_similar_dimensions >= 1:
            # Generate shared dimension(s)
            if room_similar_dimensions == 1:
                # Choose randomly whether to share x or y dimension
                share_x = np.random.choice([True, False])
                if share_x:
                    shared_dim_x = np.random.randint(low=min_room_entry_size, high=max_room_entry_size+1)
                else:
                    shared_dim_y = np.random.randint(low=min_room_entry_size, high=max_room_entry_size+1)
            elif room_similar_dimensions == 2:
                # Share both dimensions with the same value (square rooms)
                shared_dim = np.random.randint(low=min_room_entry_size, high=max_room_entry_size+1)
                shared_dim_x = shared_dim
                shared_dim_y = shared_dim
            elif room_similar_dimensions == 3:
                # Share both dimensions but with different values
                shared_dim_x = np.random.randint(low=min_room_entry_size, high=max_room_entry_size+1)
                shared_dim_y = np.random.randint(low=min_room_entry_size, high=max_room_entry_size+1)
        
        ### Base matrix
        base_matrix = np.zeros(grid_dims)
        room_n = 1
        for i in range(base_matrix.shape[0]):
            for j in range(base_matrix.shape[1]):
                if base_matrix[i,j] == 0.:
                    aux_col = np.where(base_matrix[i:,j] != 0)[0]
                    aux_row = np.where(base_matrix[i,j:] != 0)[0]
                    if len(aux_col) != 0:
                        remaining_x = aux_col[0]
                    else:
                        remaining_x = len(base_matrix[i:,j])
                    if len(aux_row) != 0:
                        remaining_y = aux_row[0]
                    else:
                        remaining_y = len(base_matrix[i,j:])
                    remaining = [remaining_x, remaining_y]
                    
                    # Generate room dimensions based on symmetry settings
                    if room_similar_dimensions == 0:
                        # Original behavior - random dimensions for each room
                        room_entry_size = [min(remaining[0], np.random.randint(low=min_room_entry_size, high=max_room_entry_size+1, size=(1))[0]),\
                                           min(remaining[1], np.random.randint(low=min_room_entry_size, high=max_room_entry_size+1, size=(1))[0])]
                    else:
                        # Use shared dimensions where applicable - only create room if it fits
                        if shared_dim_x is not None:
                            dim_x = shared_dim_x if remaining[0] >= shared_dim_x else None
                        else:
                            dim_x = min(remaining[0], np.random.randint(low=min_room_entry_size, high=max_room_entry_size+1, size=(1))[0])
                        
                        if shared_dim_y is not None:
                            dim_y = shared_dim_y if remaining[1] >= shared_dim_y else None
                        else:
                            dim_y = min(remaining[1], np.random.randint(low=min_room_entry_size, high=max_room_entry_size+1, size=(1))[0])
                        
                        # Only create room if both dimensions fit (or are not constrained by symmetry)
                        if dim_x is not None and dim_y is not None:
                            room_entry_size = [dim_x, dim_y]
                        else:
                            room_entry_size = [1, 1]  # Will be marked as -1 (no room) due to size constraint

                    if (room_entry_size[0] >= min_room_entry_size) & (room_entry_size[1] >= min_room_entry_size):
                        room_id = room_n
                        room_n += 1
                    else:
                        room_id = -1
                    for ii in range(room_entry_size[0]):
                        for jj in range(room_entry_size[1]):
                            base_matrix[i+ii, j+jj] = room_id
        self.max_n_rooms = max(self.max_n_rooms, room_n)
        
        # Apply global symmetries if enabled
        global_level = base_graph_settings.get("symmetries", {}).get("global_level", 0)
        if global_level > 0:
            base_matrix = self.apply_global_symmetries(base_matrix, global_level)
        
        return base_matrix

    def apply_global_symmetries(self, base_matrix, global_level):
        """Apply global symmetries and ensure unique room IDs."""
        if global_level == 0:
            return base_matrix  # No symmetries applied
        
        # Ensure integer dtype throughout the process
        import numpy as np
        base_matrix = base_matrix.astype(int) 
        original_matrix = base_matrix.copy()
        
        if global_level == 1:
            # Single axis mirror - randomly choose direction
            use_horizontal = np.random.choice([True, False])
            if use_horizontal:
                # Horizontal mirror (left-right)
                half_width = base_matrix.shape[1] // 2
                if half_width > 0:
                    # Clear the target region first to avoid fragments
                    base_matrix[:, half_width:] = 0
                    # Apply symmetry
                    target_width = base_matrix.shape[1] - half_width
                    source_region = original_matrix[:, :half_width]
                    if target_width == half_width:
                        # Even width - perfect mirror
                        base_matrix[:, half_width:] = np.fliplr(source_region)
                    else:
                        # Odd width - mirror the maximum possible
                        mirrored = np.fliplr(source_region)
                        base_matrix[:, -half_width:] = mirrored
            else:
                # Vertical mirror (top-bottom) 
                half_height = base_matrix.shape[0] // 2
                if half_height > 0:
                    # Clear the target region first
                    base_matrix[half_height:, :] = 0
                    # Apply symmetry
                    target_height = base_matrix.shape[0] - half_height
                    source_region = original_matrix[:half_height, :]
                    if target_height == half_height:
                        # Even height - perfect mirror
                        base_matrix[half_height:, :] = np.flipud(source_region)
                    else:
                        # Odd height - mirror the maximum possible
                        mirrored = np.flipud(source_region)
                        base_matrix[-half_height:, :] = mirrored
                
        elif global_level == 2:
            # Quadrant symmetry
            half_height = base_matrix.shape[0] // 2
            half_width = base_matrix.shape[1] // 2
            
            if half_height > 0 and half_width > 0:
                # Clear all target regions first to avoid fragments
                base_matrix[:, half_width:] = 0  # Right half
                base_matrix[half_height:, :half_width] = 0  # Bottom left
                base_matrix[half_height:, half_width:] = 0  # Bottom right
                
                # Get source quadrant
                top_left = original_matrix[:half_height, :half_width]
                
                # Apply to all quadrants
                base_matrix[:half_height, :half_width] = top_left
                
                # Handle right quadrants
                if base_matrix.shape[1] - half_width == half_width:
                    base_matrix[:half_height, half_width:] = np.fliplr(top_left)
                else:
                    base_matrix[:half_height, -half_width:] = np.fliplr(top_left)
                
                # Handle bottom quadrants  
                if base_matrix.shape[0] - half_height == half_height:
                    base_matrix[half_height:, :half_width] = np.flipud(top_left)
                    if base_matrix.shape[1] - half_width == half_width:
                        base_matrix[half_height:, half_width:] = np.flipud(np.fliplr(top_left))
                    else:
                        base_matrix[half_height:, -half_width:] = np.flipud(np.fliplr(top_left))
                else:
                    base_matrix[-half_height:, :half_width] = np.flipud(top_left)
                    base_matrix[-half_height:, -half_width:] = np.flipud(np.fliplr(top_left))
        
        # Ensure matrix remains integer type before reassignment
        base_matrix = base_matrix.astype(int)
        
        # Reassign room IDs to ensure uniqueness
        return self.reassign_room_ids(base_matrix)
    
    def reassign_room_ids(self, base_matrix):
        """Assign unique IDs to each connected component."""
        try:
            from scipy.ndimage import label
        except ImportError:
            # Fallback without scipy
            return self.reassign_room_ids_manual(base_matrix)
        
        import numpy as np
        # Ensure integer dtype to prevent float precision issues
        base_matrix = base_matrix.astype(int)
        new_matrix = np.full_like(base_matrix, -1, dtype=int)
        current_id = 1
        
        # Process each unique room ID separately
        unique_room_ids = np.unique(base_matrix)
        for room_id in unique_room_ids:
            if room_id <= 0:  # Skip walls (-1) and empty spaces (0)
                continue
            
            # Create mask for this specific room ID
            room_mask = (base_matrix == room_id).astype(int)
            
            # Find connected components for this room
            labeled, num_components = label(room_mask)
            
            # Assign new unique IDs to each component of this room
            for i in range(1, num_components + 1):
                component_mask = (labeled == i)
                new_matrix[component_mask] = current_id
                current_id += 1
        
        # Preserve walls and empty spaces with explicit values
        wall_mask = (base_matrix == -1)
        empty_mask = (base_matrix == 0)
        new_matrix[wall_mask] = -1
        new_matrix[empty_mask] = 0
        
        return new_matrix
    
    def reassign_room_ids_manual(self, base_matrix):
        """Manual room ID reassignment without scipy."""
        import numpy as np
        # Ensure integer dtype 
        base_matrix = base_matrix.astype(int)
        new_matrix = np.full_like(base_matrix, -1, dtype=int)
        current_id = 1
        
        def flood_fill(matrix, start_i, start_j, target_value):
            visited = np.zeros_like(matrix, dtype=bool)
            stack = [(start_i, start_j)]
            cells = []
            
            while stack:
                i, j = stack.pop()
                if (i < 0 or i >= matrix.shape[0] or 
                    j < 0 or j >= matrix.shape[1] or
                    visited[i, j] or matrix[i, j] != target_value):
                    continue
                
                visited[i, j] = True
                cells.append((i, j))
                
                # Add neighbors
                stack.extend([(i-1, j), (i+1, j), (i, j-1), (i, j+1)])
            
            return cells, visited
        
        # Process each unique room ID separately
        unique_room_ids = np.unique(base_matrix)
        for room_id in unique_room_ids:
            if room_id <= 0:  # Skip walls (-1) and empty spaces (0)
                continue
            
            # Create a copy to track processed cells for this room
            global_visited = np.zeros_like(base_matrix, dtype=bool)
            
            # Find connected components for this specific room ID
            for i in range(base_matrix.shape[0]):
                for j in range(base_matrix.shape[1]):
                    if (base_matrix[i, j] == room_id and not global_visited[i, j]):
                        # Found a new connected component of this room
                        cells, local_visited = flood_fill(base_matrix, i, j, room_id)
                        
                        # Mark these cells with new ID
                        for cell_i, cell_j in cells:
                            new_matrix[cell_i, cell_j] = current_id
                            global_visited[cell_i, cell_j] = True
                        
                        current_id += 1
        
        # Preserve walls and empty spaces with explicit integer values
        wall_mask = (base_matrix == -1)
        empty_mask = (base_matrix == 0)
        new_matrix[wall_mask] = -1
        new_matrix[empty_mask] = 0
        
        return new_matrix


    def generate_graph_from_base_matrix(self, base_matrix, add_noise = False, add_multiview = False):
        graph = GraphWrapper()
        room_center_distances = self.settings["source"]["base_graphs"]["room_center_distances"]
        wall_thickness = np.random.uniform(self.settings["source"]["base_graphs"]["wall_thickness"][0], self.settings["source"]["base_graphs"]["wall_thickness"][1])

        if add_noise:
            if self.settings["noise"]["global"]["active"]:
                noise_global_center = np.concatenate([np.array(self.settings["source"]["base_graphs"]["playground_size"]) * self.settings["noise"]["global"]["translation"] * (np.random.rand(2)- 0.5), [0]])
                noise_global_rotation_angle = (np.random.rand(1)*360*self.settings["noise"]["global"]["rotation"])[0]
            else:
                noise_global_center = [0,0,0]
                noise_global_rotation_angle = 0

        ### Rooms
        room_ids = np.unique(base_matrix)
        room_ids = np.delete(room_ids, np.where(room_ids == -1))
        room_id_to_node_id = {}  # Mapping from base_matrix room_id to graph node_id
        
        for base_matrix_room_id in room_ids:
            occurrencies = np.argwhere(np.where(base_matrix == base_matrix_room_id, True, False))
            limits = [occurrencies[0],occurrencies[-1]]
            room_entry_size = [limits[1][0] - limits[0][0] + 1, limits[1][1] - limits[0][1] + 1]
            node_ID = max(graph.get_nodes_ids(), default=-1) + 1
            room_id_to_node_id[base_matrix_room_id] = node_ID  # Store mapping
            room_center = np.array([room_center_distances[0]*(limits[0][0] + (room_entry_size[0]-1)/2), room_center_distances[1]*(limits[0][1]+(room_entry_size[1]-1)/2), 0])
            room_orientation_angle = 0.0
            room_area = [room_center_distances[0]*room_entry_size[0] - wall_thickness/2, room_center_distances[1]*room_entry_size[1] - wall_thickness/2, 0]
            if add_noise:
                if self.settings["noise"]["global"]["active"]:
                    room_orientation_angle += noise_global_rotation_angle

                if self.settings["noise"]["room"]["active"]:
                    center_noise = np.concatenate([np.random.rand(2)*room_center_distances*self.settings["noise"]["room"]["translation"], [0]])
                    room_orientation_angle += (np.random.rand(1)-0.5)[0]*360*self.settings["noise"]["room"]["rotation"]
                else:
                    center_noise = [0,0,0]
                
                room_center = R.from_euler("Z", noise_global_rotation_angle, degrees= True).apply(np.array(noise_global_center) + np.array(room_center) + center_noise)
                # room_area = abs(R.from_euler("Z", room_orientation_angle, degrees= True).apply(room_area))
            geometric_info = room_center
            viz_center = room_center + self.viz_center_offsets["room"]

            room_viz = copy.deepcopy(viz_data_base)
            room_viz.update({"type": "Point", "feat": 'ro', "center": viz_center, "linewidth": 1, "alpha": 1.0, "size": 1})

            graph.add_nodes([(node_ID,{"type" : "room","center" : room_center, "x": room_center, "orientation_angle": room_orientation_angle, "area" : room_area, "Geometric_info" : geometric_info,\
                                            "viz" : room_viz})])
        if add_multiview:
            num_multiviews = self.settings["multiview"]["number"]
            overlapping = self.settings["multiview"]["overlapping"]
            all_node_ids = graph.get_nodes_ids()
            masks = []
            for view_id in range(1, num_multiviews + 1):
                frontier = [int(len(all_node_ids)*(view_id-1)/ num_multiviews - np.random.randint(overlapping)),\
                    int(len(all_node_ids)* view_id / num_multiviews + np.random.randint(overlapping))]
                mask = [True if i in list(range(frontier[0], frontier[1])) else False for i in range(len(all_node_ids))]
                masks.append(mask)
            masks = np.array(masks)
            for i, node_id in enumerate(list(graph.get_nodes_ids())):
                graph.update_node_attrs(node_id, {"view" : np.squeeze(np.argwhere(masks[:, i]), axis= 1)+1})

        ### Wall surfaces
        room_nodes_data = copy.deepcopy(graph.get_attributes_of_all_nodes())
        canonic_normals = [[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]]

        
        for node_data in room_nodes_data:
            normals = copy.deepcopy(canonic_normals)
            if add_noise:
                if self.settings["noise"]["ws"]["active"]:
                    per_ws_noise_rot_angle = (np.random.rand(4)-np.ones(4)*0.5) * 360 * self.settings["noise"]["ws"]["rotation"]
                else:
                    per_ws_noise_rot_angle = [0,0,0,0]
                normals = np.array([list(R.from_euler("Z", node_data[1]["orientation_angle"] + per_ws_noise_rot_angle[j], degrees= True).apply(normals[j])) for j in range(4)])
                
            for i in range(4):
                feature_dict = {}
                node_ID = max(graph.get_nodes_ids(), default=-1) + 1
                orthogonal_normal = R.from_euler("Z", 90, degrees= True).apply(copy.deepcopy(normals[i]))
                orthogonal_canonic_normal = R.from_euler("Z", 90, degrees= True).apply(copy.deepcopy(canonic_normals[i]))
                ws_normal = np.array([-1,-1, 0], dtype=np.float64)*normals[i] ### DBG FLAG
                ws_center = node_data[1]["center"] + abs(np.dot(np.array(node_data[1]['area'])/2,canonic_normals[i]))*np.array(normals[i])

                ws_length = abs(np.dot(np.array(node_data[1]['area']),canonic_normals[i]))
                ws_limit_1 = ws_center + abs(np.dot(np.array(node_data[1]['area'])/2,np.array(orthogonal_canonic_normal)))*np.array(orthogonal_normal)
                ws_limit_2 = ws_center + abs(np.dot(np.array(node_data[1]['area'])/2,-np.array(orthogonal_canonic_normal)))*(-np.array(orthogonal_normal))
                ws_length = np.linalg.norm(ws_limit_1 - ws_limit_2)
                
                # feature_dict = {"ws_center": ws_center, "ws_normal": ws_normal, "ws_length": ws_length}
                # embedding_builder = NodeEdgeFeatureEmbeddingBuildier("node", feature_dict)
                # x = embedding_builder.build_embedding(self.settings["initial_features"]["nodes"]["ws"])

                y = int(node_data[0])
                geometric_info = np.concatenate([ws_center, ws_normal])
                color_map = ["green", "orange", "red", "pink"]
                color_map = ["black", "black", "black", "black"]

                ws_viz = copy.deepcopy(viz_data_base)
                ws_viz.update({"type": "Line", "feat": color_map[i], "limits" : [ws_limit_1,ws_limit_2],"center": ws_center, "linewidth": 2.0, "alpha": 0.5, "size": 1})

                graph.add_nodes([(node_ID,{"type" : "ws","center" : ws_center, "y" : y, "normal" : ws_normal, "Geometric_info" : geometric_info,\
                                           "canonic_normal_index" : canonic_normals[i], "linewidth": 2.0, "limits": [ws_limit_1,ws_limit_2],
                                           "length": ws_length, "viz" : ws_viz})])
                graph.add_edges([(node_ID, node_data[0], {"type": "ws_belongs_room", "x": [], "viz_feat" : 'red', "linewidth":1.0, "alpha":0.5})])
                
                ### Fully connected version
                for prior_ws_i in range(i):
                    x = segments_distance(graph.get_attributes_of_node(node_ID)["limits"],graph.get_attributes_of_node(node_ID-(prior_ws_i+1))["limits"])
                    graph.add_edges([(node_ID, node_ID-(prior_ws_i+1), {"type": "ws_same_room", "x":x, "viz_feat": "b", "linewidth":1.0, "alpha":0.5})])

                if add_multiview:
                    graph.update_node_attrs(node_ID, {"view" : graph.get_attributes_of_node(node_data[0])["view"]})


        ### Walls
        explored_walls = []
        for i in range(base_matrix.shape[0]):
            for j in range(base_matrix.shape[1]):
                for ij_difference in [[1,0], [0,1]]:
                    ij_difference_3D = ij_difference + [0]
                    compared_ij = [i + ij_difference[0], j + ij_difference[1]]
                    current_room_id = base_matrix[i,j]
                    comparison = np.array(base_matrix.shape) > np.array(compared_ij)
                    if current_room_id > 0 and comparison.all() and current_room_id != base_matrix[compared_ij[0],compared_ij[1]]:
                        compared_room_id = base_matrix[compared_ij[0],compared_ij[1]]
                        if compared_room_id > 0 and (current_room_id, compared_room_id) not in explored_walls:
                            explored_walls.append((current_room_id, compared_room_id))
                            graph.to_directed()
                            # Use mapping to get correct node IDs
                            current_node_id = room_id_to_node_id[current_room_id]
                            compared_node_id = room_id_to_node_id[compared_room_id]
                            
                            current_room_neigh = graph.get_neighbourhood_graph(current_node_id).filter_graph_by_node_types(["ws"])
                            current_room_neigh_ws_id = list(current_room_neigh.filter_graph_by_node_attributes({"canonic_normal_index" : ij_difference_3D}).get_nodes_ids())[0]
                            current_room_neigh_ws_center = current_room_neigh.get_attributes_of_node(current_room_neigh_ws_id)["center"]

                            compared_room_neigh = graph.get_neighbourhood_graph(compared_node_id).filter_graph_by_node_types(["ws"])
                            compared_room_neigh = graph.get_neighbourhood_graph(compared_node_id).filter_graph_by_node_types(["ws"])
                            ij_difference_3D_oppposite = list(-1*np.array(ij_difference_3D))
                            compared_room_neigh_ws_id = list(compared_room_neigh.filter_graph_by_node_attributes({"canonic_normal_index" : ij_difference_3D_oppposite}).get_nodes_ids())[0]
                            compared_room_neigh_ws_center = compared_room_neigh.get_attributes_of_node(compared_room_neigh_ws_id)["center"]

                            wall_center = np.array(np.array(current_room_neigh_ws_center) + (np.array(compared_room_neigh_ws_center) - np.array(current_room_neigh_ws_center))/2)
                            viz_wall_center = wall_center + self.viz_center_offsets["wall"]
                            node_ID = max(graph.get_nodes_ids(), default=-1) + 1

                            wall_viz = copy.deepcopy(viz_data_base)
                            wall_viz.update({"type": "Point", "feat": "oo", "limit" : [ws_limit_1,ws_limit_2],"center": viz_wall_center})

                            graph.add_nodes([(node_ID,{"type" : "wall", "x" : wall_center, "center" : wall_center, "viz" : wall_viz})])
                            graph.add_edges([(current_room_neigh_ws_id, node_ID, {"type": "ws_belongs_wall", "x": [], "viz_feat": "m", "linewidth":1.0, "alpha":0.5}),\
                                             (compared_room_neigh_ws_id, node_ID, {"type": "ws_belongs_wall","viz_feat": "m", "x": [], "linewidth":1.0, "alpha":0.5})])
                            graph.add_edges([(current_room_neigh_ws_id, compared_room_neigh_ws_id, {"type": "ws_same_wall", "x": [], "viz_feat": "orange", "linewidth":1.0, "alpha":0.5})])
                            if add_multiview:
                                graph.update_node_attrs(node_ID, {"view" : graph.get_attributes_of_node(current_room_neigh_ws_id)["view"]})


        return graph
    
    def add_floor_node(self, graph, z_cluster: float = 0.5):
        rooms_attrs = graph.filter_graph_by_node_attributes({"type" : "room"}).get_attributes_of_all_nodes()

        # Maintain hierarchy. if no room exists, floor cannot exist
        if not rooms_attrs:
            if self.logger:
                self.logger.info("Hierarchy Check: Skipping 'add_floor_nodes'. No node of type 'room' found to support node of type 'floor'")
            return graph
        
        # skip if floor already exists
        existing_floors = list(graph.filter_graph_by_node_types(["floor"]).get_nodes_ids())
        if existing_floors:
            return graph
        
        # cluster rooms by z (story) to avoid floor node connected to all rooms
        clusters: dict[int, list[tuple[int, dict]]] = {}
        for rid, r_attrs in rooms_attrs:
            c = np.asarray(r_attrs.get("center", [0.0, 0.0, 0.0]), dtype=float)
            if c.shape[0] == 2:
                c = np.append(c, 0.0)

            z = float(c[2])
            z_bin = int(round(z/max(z_cluster, 1e-6)))
            clusters.setdefault(z_bin, []).append((rid, r_attrs))
        
        next_id = max(graph.get_nodes_ids(), default=-1) + 1
        
        for z_bin, members in sorted(clusters.items(), key=lambda kv: kv[0]):
            room_centers = [np.asarray(m[1].get("center", [0.0, 0.0, 0.0]), dtype=float) for m in members]
            room_centers = [c if c.shape[0] == 3 else np.append(c, 0.0) for c in room_centers]
            floor_center = np.mean(np.array(room_centers), axis=0)
            
            floor_node_id = next_id
            next_id += 1
            
            viz_floor_center = floor_center + self.viz_center_offsets["floor"]
            floor_viz = copy.deepcopy(viz_data_base)
            floor_viz.update({"type": "Point", "feat": "go", "center": viz_floor_center})
            
            graph.add_nodes([(floor_node_id, {"type": "floor", "x": floor_center, "center": floor_center, "viz": floor_viz})])
            
            for rid, _ in members:
                graph.add_edges([(rid, floor_node_id, {
                    "type": "room_belongs_floor",
                    "x": [],
                    "viz_feat": "g",
                    "linewidth": 1.0,
                    "alpha": 0.5
                })])
        
        """
        # calc center strictly from room
        room_centers = [attr[1]["center"] for attr in rooms_attrs]
        floor_center = np.mean(np.array(room_centers), axis=0)

        floor_node_id = max(graph.get_nodes_ids(), default=-1) + 1

        viz_floor_center = floor_center + self.viz_center_offsets["floor"]
        floor_viz = copy.deepcopy(viz_data_base)
        floor_viz.update({"type": "Point", "feat": "go","center": viz_floor_center})

        # add node at center   
        graph.add_nodes([(floor_node_id,{"type" : "floor", "x" : floor_center, "center" : floor_center, "viz" : floor_viz})])

        # connect rooms to floor if rooms exist
        room_ids = [attr[0] for attr in rooms_attrs]
        for room_id in room_ids:
            graph.add_edges([(room_id, floor_node_id, {"type": "room_belongs_floor", "x": [],"viz_feat": "g",\
                                                        "linewidth":1.0, "alpha":0.5})])
        """
        
        return graph   
    
    def add_building_node(self, graph):
        # dont create building node if it already exists:
        existing_buildings = list(graph.filter_graph_by_node_types(["building"]).get_nodes_ids())
        if existing_buildings:
            floors_attrs = graph.filter_graph_by_node_attributes({"type" : "floor"}).get_attributes_of_all_nodes()
            if floors_attrs:
                floor_centers = np.array([attr[1]["center"] for attr in floors_attrs])
                avg_xy = floor_centers[:, :2].mean(axis=0)  
                max_z = floor_centers[:, 2].max()
                bn_offset = 2.0
                building_center = np.array([avg_xy[0], avg_xy[1], max_z + bn_offset])

                bid = existing_buildings[0]
                b_attrs = graph.get_attributes_of_node(bid)
                b_attrs["center"] = building_center
                b_attrs["x"] = building_center

                viz = b_attrs.get("viz", {})
                if isinstance(viz, dict):
                    viz_building_center = building_center + self.viz_center_offsets["building"]
                    viz["center"] = viz_building_center
                    b_attrs["viz"] = viz

                graph.update_node_attrs(bid, b_attrs)

            return graph

        floors_attrs = graph.filter_graph_by_node_attributes({"type" : "floor"}).get_attributes_of_all_nodes()

        # Fallback: Handle no floor node existing
        if not floors_attrs:
            if self.logger:
                self.logger.info("Hierarchy Check: Skipping 'add_building_nodes'. No node of type 'floor' found to support node of type 'building'")
            return graph

        floor_centers = np.array([attr[1]["center"] for attr in floors_attrs])

        # calc XY as mean but Z as MAX + offset
        avg_xy = floor_centers[:, :2].mean(axis=0)  
        max_z = floor_centers[:, 2].max()

        bn_offset = 2.0
        building_center = np.array([avg_xy[0], avg_xy[1], max_z + bn_offset])

        building_node_id = max(graph.get_nodes_ids()) + 1

        viz_building_center = building_center + self.viz_center_offsets["building"]
        building_viz = copy.deepcopy(viz_data_base)
        building_viz.update({"type": "Point", "feat": "co","center": viz_building_center})

        graph.add_nodes([(building_node_id,{"type" : "building", "x" : building_center, "center" : building_center, "viz" : building_viz})])

        # Specifically connect Floor only to Building
        floor_ids = [attr[0] for attr in floors_attrs]
        for floor_id in floor_ids:
            graph.add_edges([(floor_id, building_node_id, {"type": "floor_belongs_building", "x": [],"viz_feat": "c",\
                                                        "linewidth":1.0, "alpha":0.5})])
             
        return graph
    
    def add_stories(self, graph, n_floors = None, add_floor_nodes = False):
        story_height = 5
        initial_graph = copy.deepcopy(graph)

        if add_floor_nodes:
            initial_graph = self.add_floor_node(initial_graph)

        working_graph = copy.deepcopy(initial_graph)
        for n_floor in range(n_floors - 1):
            new_graph = copy.deepcopy(initial_graph)

            #if add_floor_nodes:
            #    new_graph = self.add_floor_node(new_graph)

            current_story_height = story_height * (n_floor + 1)
            new_graph.translate_geometries(np.array([0,0,current_story_height]))

            id_offset = max(working_graph.get_nodes_ids()) + 1
            id_mapping = {o: i + id_offset for i, o in enumerate(new_graph.get_nodes_ids())}
            new_graph.relabel_nodes(mapping=id_mapping, copy=False)
            # print(f"dbg graph {graph.get_nodes_ids()}")
            # print(f"dbg id_mapping {id_mapping}")
            working_graph = working_graph.merge_graph(new_graph)

        working_graph = self.add_building_node(working_graph)

        return working_graph
    
    def add_buildings(self, graph, n_buildings = None, area_shape = None, area_radius = None):
        """Function to add additional buildings in a given area shape, within a given area radius"""
        n_buildings_default = 5
        area_shape_default = "circular"
        area_radius_default = 50.0
        z_min_offset, z_max_offset = -2.0, 2.0  # offset to generate buildings within that range

        # n_buildings is max range [1, n_buildings]
        if n_buildings is None or n_buildings < 1:
            n_buildings = n_buildings_default # default 
            if self.logger:
                self.logger.warning(f"n_buildings not defined or less than 1, applying default n_buildings = {n_buildings}")

        n_extra = random.randint(1, int(n_buildings))

        # normalize area shape
        effective_area_shape = area_shape.lower()  # type: ignore
        if effective_area_shape is None or effective_area_shape not in ("circular", "square"):
            effective_area_shape = area_shape_default # default
            if self.logger:
                self.logger.warning(f"Invalid area_shape = '{area_shape}'. Defaulting to 'circular'")

        # area fallback
        effective_area_radius = area_radius
        if effective_area_radius is None:
            effective_area_radius = area_radius_default # default 
            if self.logger:
                self.logger.info(
                    f"area_radius not defined, using default: {effective_area_radius:.2f}"
                )

        # establish nodes to mimic the exact config of the original building
        # this is to enable add_buildings to generate additional buildings even
        # if the config generates a graph of only ws nodes
        base_node_types = set()
        for _, attrs in graph.get_attributes_of_all_nodes():
            base_node_types.add(attrs.get("type"))

        # empty graph handling
        if not base_node_types:
            base_node_types  = {"ws", "wall", "room", "floor", "building"}  # hierarchy fallback

        # story setting helper
        def _sample_config_n_stories():
            max_n = 1
            try:
                pp = self.settings.get("postprocess", {})
                if isinstance(pp, dict):
                    for config, pp_list in pp.items():
                        # only include add_buildings 
                        if any(isinstance(p, dict) and p.get("pp_name") == "add_buildings" for p in pp_list):
                            # look for add_stories
                            for p in pp_list:
                                if isinstance(p, dict) and p.get("pp_name") == "add_stories":
                                    max_n = int(p.get("n_stories", 1))
            except Exception:
                pass
            max_n = max(1, max_n)

            # propagate and return array with random story distributions 
            return[random.randint(1, max_n) for _ in range(n_extra)]

        # additional story helpers
        def _bbox_xy(bbox):
            (minx, miny, *_), (maxx, maxy, *_) = bbox
            return float(minx), float(miny), float(maxx), float(maxy)

        def _uniform_scale_xy_about(gw, scale, about_xy):
            """Uniformly scale all node geometries in XY about a given pivot."""
            if abs(scale - 1.0) < 1e-9:
                return
            about_xy = np.asarray(about_xy[:2], dtype=float)

            def _scale_pt(p):
                p = np.asarray(p, dtype=float)
                if p.shape[0] == 2:
                    xy, z = p, 0.0
                else:
                    xy, z = p[:2], p[2]
                xy = (xy - about_xy) * scale + about_xy
                return np.array([xy[0], xy[1], z], dtype=float)

            for nid, attrs in gw.get_attributes_of_all_nodes():
                viz = attrs.get("viz", {})

                if "center" in attrs:
                    attrs["center"] = _scale_pt(attrs["center"])
                if "center" in viz:
                    viz["center"] = _scale_pt(viz["center"])
                    attrs["viz"] = viz

                if "limits" in attrs and isinstance(attrs["limits"], (list, tuple)) and len(attrs["limits"]) == 2:
                    a, b = attrs["limits"]
                    attrs["limits"] = [_scale_pt(a), _scale_pt(b)]
                if "limits" in viz and isinstance(viz["limits"], (list, tuple)) and len(viz["limits"]) == 2:
                    a, b = viz["limits"]
                    viz["limits"] = [_scale_pt(a), _scale_pt(b)]
                    attrs["viz"] = viz

        def _translate_xy(gw, delta_xy):
            """Translate all node geometries in XY by delta_xy."""
            delta_xy = np.asarray(delta_xy[:2], dtype=float)

            def _shift_pt(p):
                p = np.asarray(p, dtype=float)
                if p.shape[0] == 2:
                    xy, z = p, 0.0
                else:
                    xy, z = p[:2], p[2]
                xy = xy + delta_xy
                return np.array([xy[0], xy[1], z], dtype=float)

            for nid, attrs in gw.get_attributes_of_all_nodes():
                viz = attrs.get("viz", {})

                if "center" in attrs:
                    attrs["center"] = _shift_pt(attrs["center"])
                if "center" in viz:
                    viz["center"] = _shift_pt(viz["center"])
                    attrs["viz"] = viz

                if "limits" in attrs and isinstance(attrs["limits"], (list, tuple)) and len(attrs["limits"]) == 2:
                    a, b = attrs["limits"]
                    attrs["limits"] = [_shift_pt(a), _shift_pt(b)]
                if "limits" in viz and isinstance(viz["limits"], (list, tuple)) and len(viz["limits"]) == 2:
                    a, b = viz["limits"]
                    viz["limits"] = [_shift_pt(a), _shift_pt(b)]
                    attrs["viz"] = viz

        def _fit_floor_to_base_bbox(floor_gw, base_bbox):
            """
            Uniformly scale + align the floor so its XY AABB fits inside the base floor AABB.
            Never scale up above 1.0 (floors can be smaller, not larger).
            TODO: if time, make floors able to be larger than base to a certain threshold
            """
            # target (base) box
            bminx, bminy, bmaxx, bmaxy = _bbox_xy(base_bbox)
            tw, th = max(1e-9, bmaxx - bminx), max(1e-9, bmaxy - bminy)
            tcenter = np.array([(bminx + bmaxx) / 2.0, (bminy + bmaxy) / 2.0], dtype=float)

            # source (floor) box
            fminx, fminy, fmaxx, fmaxy = _bbox_xy(floor_gw.get_bounding_box())
            fw, fh = max(1e-9, fmaxx - fminx), max(1e-9, fmaxy - fminy)
            fcenter = np.array([(fminx + fmaxx) / 2.0, (fminy + fmaxy) / 2.0], dtype=float)

            # scale: fit inside target, but do not enlarge above 1.0
            s_fit = min(tw / fw, th / fh)
            s = min(1.0, s_fit)
            _uniform_scale_xy_about(floor_gw, s, about_xy=fcenter)

            # re-center to target center
            _translate_xy(floor_gw, (tcenter - fcenter))


        # rotation helper functions 
        def _building_pivot(graph):
            """pivot from building node (to ensure different floors are rotated identically as the base floor per building)"""
            building_nodes = list(graph.filter_graph_by_node_types(["building"]).get_nodes_ids())
            if building_nodes:
                building_attrs = graph.get_attributes_of_node(building_nodes[0]) or {}
                center = None

                # prefer top-level center, fallback viz center, else origin
                if "center" in building_attrs:
                    center = np.asarray(building_attrs["center"], dtype=float)
                elif "viz" in building_attrs and isinstance(building_attrs["viz"], dict) and "center" in building_attrs["viz"]:
                    center = np.asarray(building_attrs["viz"]["center"], dtype=float)
                if center is not None:
                    if center.shape[0] == 2:
                        center = np.append(center, 0.0)
                    return center[:3]
                
                # fallback
                center = graph.get_graph_center().astype(float, copy=False)
                if center.shape[0] == 2:
                    center = np.append(center, 0.0)
                return center[:3]
        
        def _apply_rigid_z_rotation(graph, angle_rad, pivot_xyz, delta_xyz):
            """
            applies rigid z rotation for each building such that only the base is rotated
            and the floors above mimic that rotation to prevent different rotations
            for each individual floor
            """
            cos, sin = np.cos(angle_rad), np.sin(angle_rad)
            R = np.array([[cos, -sin, 0.0],
                        [sin, cos, 0.0],
                        [0.0, 0.0, 1.0]], dtype=float)
            
            pivot = np.asarray(pivot_xyz, dtype=float)
            if pivot.shape[0] == 2:
                pivot = np.append(pivot, 0.0)
            pivot = pivot[:3]

            delta = np.asarray(delta_xyz, dtype=float)
            if delta.shape[0] == 2:
                delta = np.append(delta, 0.0)
            delta = delta[:3]

            def _rigid_point(p):
                p = np.asarray(p, dtype=float)
                if p.shape[0] == 2:
                    p = np.append(p, 0.0)
                p = p[:3]
                return (R @ (p - pivot)) + pivot + delta

            def _rot_normal(n):
                n = np.asarray(n, dtype=float)
                if n.shape[0] == 2:
                    n = np.append(n, 0.0)
                n = n[:3]
                return (R @ n)

            all_attrs = graph.get_attributes_of_all_nodes()
            for nid, attrs in all_attrs:
                viz = attrs.get("viz", {})

                # centers
                if "center" in attrs:
                    attrs["center"] = _rigid_point(attrs["center"])
                if "center" in viz:
                    viz["center"] = _rigid_point(viz["center"])
                    attrs["viz"] = viz

                # limits (two endpoints)
                if "limits" in attrs and isinstance(attrs["limits"], (list, tuple)) and len(attrs["limits"]) == 2:
                    a, b = attrs["limits"]
                    attrs["limits"] = [_rigid_point(a), _rigid_point(b)]
                if "limits" in viz and isinstance(viz["limits"], (list, tuple)) and len(viz["limits"]) == 2:
                    a, b = viz["limits"]
                    viz["limits"] = [_rigid_point(a), _rigid_point(b)]
                    attrs["viz"] = viz

                # normals
                if "normal" in attrs:
                    attrs["normal"] = _rot_normal(attrs["normal"])

        def _aabb2d_overlap(b1, b2, pad=0.0):
            """Returns True if two 2D AABBs (XY only) overlap when each is expanded by padding"""
            min1 = np.asarray(b1[0], float)[:2] - pad
            max1 = np.asarray(b1[1], float)[:2] + pad
            min2 = np.asarray(b2[0], float)[:2] - pad
            max2 = np.asarray(b2[1], float)[:2] + pad
            # Non-overlap if separated on any axis; otherwise overlap
            sep = (max1[0] < min2[0]) or (max2[0] < min1[0]) or (max1[1] < min2[1]) or (max2[1] < min1[1])
            return not sep

        def _sample_non_overlapping_pose(candidate_gw, placed_bboxes, area_shape, area_radius, safety_dist=2.0, max_tries=200):
            """
            Sample a (random angle_rad, tranlsation_vector, bbox_after) for a building copy
            such that its AABB (after ridig z transformation) does not overlap in (XY)
            any bbox in "placed_bboxes" by at least "safety_dist"

            Returns (angle_rad, translation_vector, bbox_after) or None if no pose found
            """
            def _random_orientation():
                """Random angle for Z-rotation"""
                return 2.0 * np.pi * random.random()
            
            def _sample_position():
                """Return a random (dx, dy) displacement within the defined area"""
                R = float(effective_area_radius)
                if effective_area_shape == "circular":
                    # circle: uniform over disk -> r = R*sqrt(u), theta ~ U[0, 2Pi)
                    u = random.random()
                    r = R * np.sqrt(u)
                    theta = 2.0 * np.pi * random.random()
                    return np.array([r * np.cos(theta), r * np.sin(theta), 0.0], dtype=float)
                else:
                    # square: uniform in [-R, R] x [-R, R]
                    return np.array([
                        random.uniform(-R, R),
                        random.uniform(-R, R),
                        0.0
                    ], dtype=float)

            for _ in range(max_tries):
                angle = _random_orientation()
                disp  = _sample_position()

                # Build a temp copy, rigid-transform it, then test its bbox
                temp = copy.deepcopy(candidate_gw)
                pivot = _building_pivot(temp)

                # Place pivot at base_center + disp: delta = (base_center + disp) - pivot
                # base_center = base_building_gw.get_graph_center().astype(float, copy=False)
                # if base_center.shape[0] == 2: base_center = np.append(base_center, 0.0)
                delta = (anchor_z + disp) - pivot

                _apply_rigid_z_rotation(temp, angle, pivot, delta)
                bbox_new = temp.get_bounding_box()

                # Check against all placed bboxes with XY padding = safety_dist per bbox
                overlaps = any(_aabb2d_overlap(bbox_new, b, pad=safety_dist) for b in placed_bboxes)
                if not overlaps:
                    return angle, delta, bbox_new

            return None  # give up after max_tries
        
        # Store initial building graph
        combined_city_graph = copy.deepcopy(graph)

        if "floor" in base_node_types:
            has_floor = any(attrs.get("type") == "floor" for _, attrs in combined_city_graph.get_attributes_of_all_nodes())
            if not has_floor:
                combined_city_graph = self.add_floor_node(combined_city_graph)
        if "building" in base_node_types:
            combined_city_graph = self.add_building_node(combined_city_graph)

        # Get exisiting building node of the first building
        building_nodes_original = list(combined_city_graph.filter_graph_by_node_types(["building"]).get_nodes_ids())
        all_building_nodes_ids = []
        if building_nodes_original:
            all_building_nodes_ids.append(building_nodes_original[0])

        # Reference center of base building to place additional buildings around it
        base_center = graph.get_graph_center().astype(float, copy=False)

        # ground anchor to prevent floating buildings
        anchor_z = np.array([base_center[0], base_center[1], 0.0], dtype=float)

        # Keep track of all building bounding boxes to avoid overlap when placing additional buildings
        placed_bboxes = [combined_city_graph.get_bounding_box()]

        # generate floor count array once before the building generatoin loop
        n_stories_conf = _sample_config_n_stories()
        if self.logger:
            self.logger.info(f"Generated story array: {n_stories_conf}")

        # determine config source type (msd, or synthetic)
        source_type = self.settings.get("source", {}).get("type", "synthetic")
        msd_source_graphs = self.graphs.get("original", [])

        # prepare pool of uniques to avoid duplicates
        msd_indicies_pool = []
        if source_type == "msd" and msd_source_graphs:
            msd_indicies_pool = list(range(len(msd_source_graphs)))
            random.shuffle(msd_indicies_pool)  # shuffle to pick randomly but also not duplicates

        # Generate and place additional buildings
        for target_stories in n_stories_conf:
            # determine base template (base of the building)
            if source_type == "msd":
                if not msd_source_graphs:
                    if self.logger:
                        self.logger.warning("No MSD graphs available to sample from.")
                    continue

                # check if unique buildings are stil left in pool, if not repropagete pool
                # Note: current msd dataset has 3000+ buildings, this is just to make sure
                # that if only a slice of that dataset is used, you still 
                # generate additional buildings even if duplacte ones.
                if not msd_indicies_pool:
                    msd_indicies_pool = list(range(len(msd_source_graphs)))
                    random.shuffle(msd_indicies_pool)
                    if self.logger:
                        self.logger.warning("Unique MSD builidngs exhausted, refilling pool (duplicates will occure).")

                # pop unique idx
                select_idx = msd_indicies_pool.pop()

                # debug
                if self.logger:
                    self.logger.debug(f"selected msd building index: {select_idx} (Remaining pool: {len(msd_indicies_pool)})")

                # deepcopy to ensure no modification of source
                base_template = copy.deepcopy(msd_source_graphs[select_idx])

                # ensure presence of floor node
                if "floor" not in [attrs.get("type") for _, attrs in base_template.get_attributes_of_all_nodes()]:
                    base_template = self.add_floor_node(base_template)
            else:
                # generate a new syntehtic building
                base_matrix = self.generate_base_matrix()
                base_template = self.generate_graph_from_base_matrix(base_matrix=base_matrix, add_noise=False)
                base_template = self.add_floor_node(base_template)  # forced for placement logic calculation

            # init new building object
            new_builidng = copy.deepcopy(base_template)

            # base building bbox
            base_bbox = new_builidng.get_bounding_box()

            story_height = 5  # must match add_stories()
            for k in range(1, target_stories):
                if source_type == "msd":
                    # for msd additional floors are duplicates of the base
                    floor_k = copy.deepcopy(base_template)
                else:
                    # for synthetic, generate a new random layout and scale it to keep within base floor dimensions
                    fm = self.generate_base_matrix()
                    floor_k = self.generate_graph_from_base_matrix(base_matrix=fm, add_noise=False)
                    floor_k = self.add_floor_node(floor_k)

                    # fit floor to base floor (never larger)
                    _fit_floor_to_base_bbox(floor_k, base_bbox)

                # stack at height k
                floor_k.translate_geometries(np.array([0.0, 0.0, story_height * k],dtype=float))

                # relable then merge
                existing_ids_nb = new_builidng.get_nodes_ids()
                numeric_ids_nb  = [nid for nid in existing_ids_nb if isinstance(nid, int)]
                max_id_nb       = max(numeric_ids_nb) if numeric_ids_nb else -1
                id_offset_nb    = max_id_nb + 1
                id_map_nb       = {old_id: i + id_offset_nb for i, old_id in enumerate(floor_k.get_nodes_ids())}
                floor_k.relabel_nodes(mapping=id_map_nb, copy=True)  # type: ignore
                new_builidng = new_builidng.merge_graph(floor_k)
            
            new_builidng = self.add_building_node(new_builidng)

            # Ensure building node exists
            # new_builidng = self.add_building_node(new_builidng)
            new_builidng_b_nodes = list(new_builidng.filter_graph_by_node_types(["building"]).get_nodes_ids())
            new_builidng_node_id_in_candidate = new_builidng_b_nodes[0] if new_builidng_b_nodes else None

            # force z=0 for calculation
            if new_builidng_node_id_in_candidate is not None:
                b_attrs = new_builidng.get_attributes_of_node(new_builidng_node_id_in_candidate)

                # center -> z = 0
                if "center" in b_attrs:
                    b_attrs["center"] = np.array([b_attrs["center"][0], b_attrs["center"][1], 0.0])
                new_builidng.update_node_attrs(new_builidng_node_id_in_candidate, b_attrs)

            # Attempt to sample a non-overlapping placement (angle + translation)
            pose = _sample_non_overlapping_pose(
                candidate_gw=new_builidng,                  # new building
                placed_bboxes=placed_bboxes,             # already placed ones
                area_shape=effective_area_shape,
                area_radius=effective_area_radius,
                safety_dist=2.0,                         # min spacing between buildings
                max_tries=300
            )

            if pose is None:
                if self.logger:
                    self.logger.warning("Could not find non-overlapping placement — skipping this building copy.")
                continue

            angle, delta, bbox_after = pose

            # generate building at random z within previously set bounds
            z_displacement = random.uniform(z_min_offset, z_max_offset)
            delta[2] += z_displacement

            # build the copy and apply the non-overlapping transform
            # current_building_copy = copy.deepcopy(graph)
            pivot = _building_pivot(new_builidng)
            _apply_rigid_z_rotation(new_builidng, angle, pivot, delta)

            # geometry is set, strip out nodes to match the base building
            # if base building only has WS, additional buildings should also only have WS
            nodes_to_remove = []
            for nid, attrs in new_builidng.get_attributes_of_all_nodes():
                if attrs["type"] not in base_node_types:
                    nodes_to_remove.append(nid)
            new_builidng.remove_nodes(nodes_to_remove)

            # Relabel all nodes of the current building copy to ensure unique IDs
            existing_ids = combined_city_graph.get_nodes_ids()
            numeric_ids = [nid for nid in existing_ids if isinstance(nid, int)]
            max_id_in_combined = max(numeric_ids) if numeric_ids else -1
            id_offset = max_id_in_combined + 1
            id_mapping = {old_id: k + id_offset for k, old_id in enumerate(new_builidng.get_nodes_ids())}
            new_builidng.relabel_nodes(mapping=id_mapping, copy=True)

            # Get new Id of the building node for the current copy
            if new_builidng_node_id_in_candidate is not None and "building" in base_node_types and new_builidng_node_id_in_candidate in id_mapping:
                all_building_nodes_ids.append(id_mapping[new_builidng_node_id_in_candidate])

            # ensure graph directionalities match before merging to avoid NetworkX error
            if combined_city_graph.is_directed() and not new_builidng.graph.is_directed():
                new_builidng.to_directed()
            elif not combined_city_graph.is_directed() and new_builidng.graph.is_directed():
                new_builidng.to_undirected()

            # Merge the current builing copy into the combined city graph
            combined_city_graph = combined_city_graph.merge_graph(new_builidng)
            placed_bboxes.append(bbox_after)

        if not all_building_nodes_ids:
            return combined_city_graph
        
        # place all building nodes above its own highest floor
        bn_offset = 2.0  # without offset building node would place withing heighest floor node

        nxg_city = combined_city_graph.graph
        def _neighbors_any_dir_city(nid):
            if hasattr(nxg_city, "predecessors") and hasattr(nxg_city, "successors"):
                return set(nxg_city.predecessors(nid)) | set(nxg_city.successors(nid))  # type: ignore
            return set(nxg_city.neighbors(nid))

        building_nodes = list(combined_city_graph.filter_graph_by_node_types(["building"]).get_attributes_of_all_nodes())
        for bid, b_attrs in building_nodes:
            # Find floor neighbors of this building
            floor_neighbors = []
            try:
                for nb in _neighbors_any_dir_city(bid): 
                    nb_attrs = combined_city_graph.get_attributes_of_node(nb)
                    if nb_attrs and nb_attrs.get("type") == "floor":
                        floor_neighbors.append(nb)
            except Exception:
                # if graph is directed or neighbors() fails, just skip gracefully
                continue

            if not floor_neighbors:
                continue

            # compute max Z among this buildings floors
            floor_zs = []
            for fid in floor_neighbors:
                f_attrs = combined_city_graph.get_attributes_of_node(fid)
                c = np.asarray(f_attrs.get("center", [0.0, 0.0, 0.0]), dtype=float)
                if c.shape[0] == 2:
                    c = np.append(c, 0.0)
                floor_zs.append(float(c[2]))

            if not floor_zs:
                continue

            # Update building node center
            c = np.asarray(b_attrs.get("center", [0.0, 0.0, 0.0]), dtype=float)
            if c.shape[0] == 2:
                c = np.append(c, 0.0)

            c[2] = max(floor_zs) + bn_offset
            b_attrs["center"] = c
            b_attrs["x"] = c  # keep consistent with add_building_node()

            # Update viz center accordingly
            viz = b_attrs.get("viz", {})
            if isinstance(viz, dict):
                off = np.asarray(self.viz_center_offsets["building"], dtype=float)
                viz["center"] = c + off
                b_attrs["viz"] = viz

            combined_city_graph.update_node_attrs(bid, b_attrs)

        city_node_id = max(combined_city_graph.get_nodes_ids()) + 1
        city_offset = self.viz_center_offsets["city"]

        # recompute after update
        building_nodes = list(combined_city_graph.filter_graph_by_node_types(["building"]).get_attributes_of_all_nodes())
        if building_nodes:
            b_centers = []
            b_zs = []
            for bid, b_attrs in building_nodes:
                c = np.asarray(b_attrs.get("center", [0.0, 0.0, 0.0]), dtype=float)
                if c.shape[0] == 2:
                    c = np.append(c, 0.0)
                b_centers.append(c)
                b_zs.append(float(c[2]))

            b_centers = np.array(b_centers, dtype=float)
            city_center = b_centers.mean(axis=0)
            city_center[2] = max(b_zs) + city_offset[2]
        else:
            # fallback if no buildings (shouldn't happen in normal generation)
            city_center = combined_city_graph.get_graph_center().astype(float, copy=False)
            if city_center.shape[0] == 2:
                city_center = np.append(city_center, 0.0)

        # City relevant attributes
        city_attrs = {
            "type": "city",
            "center": city_center.tolist(),
            "viz": {
                "type": "Point",
                "center": (city_center + city_offset).tolist(),
                "feat": "ko",
                "markersize": 0.5
            }
        }
        combined_city_graph.add_nodes([(city_node_id, city_attrs)])

        # Connect city node to all building nodes
        edge_batch = []
        for building_id in all_building_nodes_ids:
            if building_id is None:
                continue
            edge_batch.append((city_node_id, building_id, {"type": "contains_building", "viz_feat": "purple"}))
            edge_batch.append((building_id, city_node_id, {"type": "belongs_to_city", "viz_feat": "purple"})) # Inverse edge

        if edge_batch:
            combined_city_graph.add_edges(edge_batch)
        
        return combined_city_graph
    

    def add_random_objects(self, graph, obj_max, distrib):
        def lines_to_polygon(lines):
            """
            lines: list of line segments, each as [(x1, y1), (x2, y2)]
            Returns a shapely Polygon if the lines form a closed shape.
            """
            # Flatten all points
            all_points = []
            for line in lines:
                all_points.extend(line)
            # Remove duplicates while preserving order
            seen = set()
            ordered_points = []
            for pt in all_points:
                tpt = tuple(pt)
                if tpt not in seen:
                    ordered_points.append(tpt)
                    seen.add(tpt)
            # Ensure the polygon is closed
            if ordered_points[0] != ordered_points[-1]:
                ordered_points.append(ordered_points[0])
            # Create the polygon
            poly = Polygon(ordered_points)
            return poly
        
        def random_points_in_polygon(polygon, n, z_val):
            """
            Randomly sample n points inside a shapely Polygon.
            Returns a list of shapely Point objects.
            """
            minx, miny, maxx, maxy = polygon.bounds
            points = []
            attempts = 0
            while len(points) < n and attempts < n * 100:
                random_point = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
                if polygon.contains(random_point):
                    points.append(random_point)
                attempts += 1
            if len(points) < n:
                print(f"Warning: Only found {len(points)} points inside the polygon after {attempts} attempts.")
            
            points_list = [[point.x, point.y, float(z_val)] for point in points]
            return points_list
        
        rooms_ids = copy.deepcopy(graph.filter_graph_by_node_types("room").get_nodes_ids())
        
        for room_id in rooms_ids:
            ws_ids = graph.get_neighbourhood_graph(room_id).filter_graph_by_node_types("ws").get_nodes_ids()
            segments = []
            for ws_id in ws_ids:
                segment = graph.get_attributes_of_node(ws_id)["limits"]
                segments.append(segment)

            poly = lines_to_polygon(segments)

            room_center = np.asarray(graph.get_attributes_of_node(room_id)["center"], dtype=float)
            room_z = float(room_center[2])

            obj_poses = random_points_in_polygon(poly, random.randint(0, obj_max["room"]), z_val=room_z)

            new_edges = []
            for obj_pose in obj_poses:
                obj_id = max(graph.get_nodes_ids()) + 1
                
                obj_pose = np.asarray(obj_pose, dtype=float)
                viz_obj_pose = obj_pose + self.viz_center_offsets["object"]
                obj_viz = copy.deepcopy(viz_data_base)
                obj_viz.update({"type": "Point", "feat": 'ks', "center": viz_obj_pose})

                obj_type = random.choice(["chair", "table"])

                graph.add_nodes([(obj_id,{"type" : "object", "object_type": obj_type, "x" : obj_pose, "center" : obj_pose, "viz" : obj_viz})])
                new_edges.append((obj_id, room_id, {"type": "object_belongs_room", "x":[], "viz_feat": "black", "linewidth":1.0, "alpha":0.5}))

            if new_edges:
                graph.add_edges(new_edges)

        wall_ids = copy.deepcopy(graph.filter_graph_by_node_types("wall").get_nodes_ids())
        for wall_id in wall_ids:
            n_wall_objects = random.randint(0, obj_max["door"])
            if n_wall_objects > 0:
                wall_attrs = graph.get_attributes_of_node(wall_id)
                wall_center = np.asarray(wall_attrs["center"], dtype=float)
                
                # Get the two ws nodes connected to this wall
                wall_ws_ids = list(graph.get_neighbourhood_graph(wall_id).filter_graph_by_node_types("ws").get_nodes_ids())
                
                if len(wall_ws_ids) >= 2:
                    # Use wall center x,y coordinates with wall's z position
                    obj_pose = np.array([wall_center[0], wall_center[1], wall_center[2]], dtype=float)

                    obj_id = max(graph.get_nodes_ids()) + 1
                    
                    viz_obj_pose = obj_pose + self.viz_center_offsets["object"]
                    obj_viz = copy.deepcopy(viz_data_base)
                    obj_viz.update({"type": "Point", "feat": 'ks', "center": viz_obj_pose})

                    graph.add_nodes([(obj_id,{"type" : "object", "object_type": "door", "x" : obj_pose, "center" : obj_pose, "viz" : obj_viz})])
                    
                    # Connect door/window to the wall
                    graph.add_edges([(obj_id, wall_id, {"type": "object_belongs_wall", "x":[], "viz_feat": "black", "linewidth":1.0, "alpha":0.5})])
                    
                    # Connect door/window to the two ws nodes
                    for ws_id in wall_ws_ids[:2]:  # Take first two ws nodes
                        graph.add_edges([(obj_id, ws_id, {"type": "object_belongs_ws", "x":[], "viz_feat": "gray", "linewidth":1.0, "alpha":0.5})])

        # Add windows to ws nodes that are not associated with any wall
        all_ws_ids = copy.deepcopy(graph.filter_graph_by_node_types("ws").get_nodes_ids())
        
        # Find ws nodes that are not connected to any walls
        ws_with_walls = set()
        for wall_id in wall_ids:
            wall_ws_ids = graph.get_neighbourhood_graph(wall_id).filter_graph_by_node_types("ws").get_nodes_ids()
            ws_with_walls.update(wall_ws_ids)
        
        ws_without_walls = [ws_id for ws_id in all_ws_ids if ws_id not in ws_with_walls]
        
        for ws_id in ws_without_walls:
            n_ws_windows = random.randint(0, obj_max.get("window", 1))  # Use window limit or default to 1
            if n_ws_windows > 0:
                ws_attrs = graph.get_attributes_of_node(ws_id)
                ws_limits = ws_attrs["limits"]  # Get the segment endpoints
                ws_z = float(ws_attrs["center"][2])  # Get z coordinate from center
                
                # Create multiple window objects
                for _ in range(n_ws_windows):
                    # Sample random point along the ws segment
                    p1, p2 = np.asarray(ws_limits[0], dtype=float), np.asarray(ws_limits[1], dtype=float)
                    t = random.random()
                    obj_xy = (1 - t) * p1[:2] + t * p2[:2]
                    obj_pose = np.array([obj_xy[0], obj_xy[1], ws_z], dtype=float)

                    obj_id = max(graph.get_nodes_ids()) + 1
                    
                    viz_obj_pose = obj_pose + self.viz_center_offsets["object"]
                    obj_viz = copy.deepcopy(viz_data_base)
                    obj_viz.update({"type": "Point", "feat": 'ks', "center": viz_obj_pose})

                    graph.add_nodes([(obj_id,{"type" : "object", "object_type": "window", "x" : obj_pose, "center" : obj_pose, "viz" : obj_viz})])
                    
                    # Connect window to the ws node
                    graph.add_edges([(obj_id, ws_id, {"type": "object_belongs_ws", "x":[], "viz_feat": "blue", "linewidth":1.0, "alpha":0.5})])
            
        return graph
    
    def apply_global_noise(self, graph, settings):
        """
        Apply global noise transformation to the graph.
        
        Args:
            graph: GraphWrapper object to transform
            settings: Dictionary containing noise parameters with keys:
                - "translation": translation noise factor
                - "rotation": rotation noise factor (in degrees)
        
        Returns:
            GraphWrapper: The transformed graph
        """
        global_translation = np.array(settings["translation"]) * (np.random.rand(2) - 0.5)
        global_rotation_angle = np.random.rand() * 360 * settings["rotation"]
        rotation_matrix_2d = R.from_euler("Z", global_rotation_angle, degrees=True).as_matrix()[:2, :2]

        for node_id, node_attrs in graph.get_attributes_of_all_nodes():
            if "center" in node_attrs:
                new_center = rotation_matrix_2d @ (node_attrs["center"][:2] + global_translation)
                node_attrs["center"][:2] = new_center
                node_attrs["viz"]["center"] = new_center

            if "normal" in node_attrs:
                new_normal = rotation_matrix_2d @ node_attrs["normal"][:2]
                node_attrs["normal"][:2] = new_normal

            if "polygon" in node_attrs:
                new_polygon = [
                    rotation_matrix_2d @ (np.array(point[:2]) + global_translation)
                    for point in node_attrs["polygon"]
                ]
                node_attrs["polygon"] = [p.tolist() for p in new_polygon]
                
            if "limits" in node_attrs:
                new_limits = []
                for point in node_attrs["limits"]:
                    point_array = np.array(point)
                    # Transform only X,Y coordinates, preserve Z if it exists
                    if len(point_array) >= 2:
                        transformed_xy = rotation_matrix_2d @ (point_array[:2] + global_translation)
                        if len(point_array) >= 3:
                            # Preserve Z coordinate
                            new_point = [transformed_xy[0], transformed_xy[1], point_array[2]]
                        else:
                            # Only had X,Y coordinates
                            new_point = [transformed_xy[0], transformed_xy[1]]
                        new_limits.append(new_point)
                node_attrs["limits"] = new_limits

            graph.update_node_attrs(node_id, node_attrs)
        
        return graph
    
    def merge_edge_types(self, graph, common_edge_type):
        new_graph = copy.deepcopy(graph)
        edges_attributes = new_graph.graph.edges(data=True)
        for edge in edges_attributes:
            if not isinstance(edge[2], dict):
                print(f"Edge {edge[2]} is not a dict, skipping.")
                continue

        for edge_attributes in edges_attributes:
            source_node_id, target_node_id, edge_attrs = edge_attributes
            if edge_attrs["type"] != common_edge_type:
                new_graph.update_edge_attrs((source_node_id, target_node_id), {"type": common_edge_type, "viz_feat" : "a", "label": 0})

        return new_graph
            
    def set_dataset(self, tag, nxdata):
        self.graphs[tag] = nxdata
    
    def get_filtered_datset(self, node_types, full_edge_types):
        print(f"SyntheticDatasetGenerator: ", Fore.GREEN + "Filtering Dataset" + Fore.WHITE)
        nx_graphs = {}
        for key in self.graphs.keys():
            nx_graphs_key = []
            for base_graph in self.graphs[key]:
                filtered_graph = base_graph.filter_graph_by_node_types(node_types)
                filtered_graph.to_directed()
                filtered_graph.relabel_nodes() ### TODO What to do when Im dealing with different node types? Check tutorial
                # specific_edge_types = [e[1] for e in full_edge_types]
                filtered_graph = filtered_graph.filter_graph_by_edge_types(full_edge_types)
                filtered_graph.to_directed()
                nx_graphs_key.append(filtered_graph)
            nx_graphs[key] = nx_graphs_key

        return nx_graphs
    
    def get_filtered_graph(self, base_graph, node_types, full_edge_types):
        filtered_graph = base_graph.filter_graph_by_node_types(node_types)
        filtered_graph.to_directed()
        filtered_graph.relabel_nodes(copy=True) ### TODO What to do when Im dealing with different node types? Check tutorial
        specific_edge_types = [e[1] for e in full_edge_types]
        filtered_graph = filtered_graph.filter_graph_by_edge_types(specific_edge_types)
        filtered_graph.to_directed()
        return filtered_graph
    
    def update_node_attrs_by_hierarchy(self, node_id, working_graph):
        hierchy_types = ["ws", "room", "floor", "building"]
        node_type = working_graph.get_attributes_of_node(node_id)["type"]
        node_attrs = working_graph.get_attributes_of_node(node_id)
        below_type = hierchy_types[hierchy_types.index(node_type) - 1]

        if node_type in ["room", "floor", "building"]:
            below_nodes = working_graph.get_neighbourhood_graph(node_id).filter_graph_by_node_types([below_type]).get_nodes_ids()
            centers = [working_graph.get_attributes_of_node(below_node)["center"] for below_node in below_nodes]
            mean_center = sum(centers) / len(centers) if centers else 0
            node_attrs["center"][0], node_attrs["center"][1] = mean_center[0], mean_center[1]
            node_attrs["viz"]["center"][0], node_attrs["viz"]["center"][1] = mean_center[0], mean_center[1]

        return working_graph

    def dropout_by_hierarchy(self, node_id, working_graph, update_higher_nodes, remove_lower_nodes):
        node_type = working_graph.get_attributes_of_node(node_id)["type"]

        ### Remove nodes
        node_ids_selected = [node_id]
        if node_type == "room" and remove_lower_nodes:
            ws_node_ids = list(working_graph.get_neighbourhood_graph(node_id).filter_graph_by_node_types(["ws"]).get_nodes_ids())
            node_ids_selected = node_ids_selected + ws_node_ids
            for ws_node_id in ws_node_ids:
                wall_ws_node_ids = list(working_graph.get_neighbourhood_graph(ws_node_id).filter_graph_by_node_types(["wall"]).get_nodes_ids())
                for wall_ws_node_id in wall_ws_node_ids:
                    if len(list(working_graph.get_neighbourhood_graph(wall_ws_node_id).filter_graph_by_node_types(["ws"]).get_nodes_ids())) < 3:
                        node_ids_selected.append(wall_ws_node_id)

                obj_ws_node_ids = list(working_graph.get_neighbourhood_graph(ws_node_id).filter_graph_by_node_types(["object"]).get_nodes_ids())
                for obj_ws_node_id in obj_ws_node_ids:
                    node_ids_selected.append(obj_ws_node_id)

            obj_ws_node_ids = list(working_graph.get_neighbourhood_graph(node_id).filter_graph_by_node_types(["object"]).get_nodes_ids())
            for obj_ws_node_id in obj_ws_node_ids:
                node_ids_selected.append(obj_ws_node_id)
                
        if node_type == "ws": ### TODO FIX
            node_ids_selected.append(node_id)

        aux_graph = copy.deepcopy(working_graph)
        working_graph.remove_nodes(node_ids_selected)

        # helper function
        def node_exists(graph, nid):
            """GraphWrapper-compatible existence check"""
            try: 
                # Will raise if nid is not present
                _ = graph.get_attribute_of_node(nid)
                return True
            except Exception:
                return False

        
        # build hierarchy list only from types that actually exist in aux_graph
        try:
            available_types = set(aux_graph.get_all_node_types())
        except Exception:
            # fallback 
            available_types = set(["ws", "room", "wall", "floor", "building"])  # best effort

        full_order = ["ws", "room", "floor", "building"]
        hierarchy_types = [t for t in full_order if t in available_types]

        updating_node_id = copy.deepcopy(node_id)

        if update_higher_nodes and node_type in hierarchy_types:
            start_idx = hierarchy_types.index(node_type) + 1
            updatable_hierarchy_types = hierarchy_types[start_idx:]

            for up_type in updatable_hierarchy_types:
                candidates = list(
                    aux_graph
                        .get_neighbourhood_graph(updating_node_id)
                        .filter_graph_by_node_types([up_type])
                        .get_nodes_ids()
                )
                if not candidates:
                    if self.logger:
                        self.logger.warning(
                            f"[dropout_by_hierarchy] No neighbour of type '{up_type}' "
                            f"from node {updating_node_id} (type={node_type}). "
                            f"Stopping hierarchy update."
                        )
                    break  # gracefully stop climbing if this hop doesn't exist

                updating_node_id = candidates[0]

                # only update if the target node still exists in working_graph
                if node_exists(working_graph, updating_node_id):
                    working_graph = self.update_node_attrs_by_hierarchy(
                        updating_node_id, working_graph
                    )
                else:
                    if self.logger:
                        self.logger.warning(
                            f"[dropout_by_hierarchy] Target node {updating_node_id} "
                            f"no longer in working_graph. Skipping."
                        )
                    break

        return working_graph

    def _remove_room_subgraph(self, graph: GraphWrapper, room_id, remove_planes: bool = True):
        """
        Removes one room and all directly related lower-hierachry entities:
        - room node
        - ws nodes
        - wall nodes
        - object nodes
        - all corresponding edges
        
        :param graph: graph passed to deconstruct
        :param room_id: room id of room to be deleted
        """
        # fast existance check
        existing = set(graph.get_nodes_ids())
        if room_id not in existing:
            return graph
        
        nodes_to_remove = {room_id}

        nxg = graph.graph
        def _neighbors_any_dir_local(nid):
            if hasattr(nxg, "predecessors") and hasattr(nxg, "successors"):
                return set(nxg.predecessors(nid)) | set(nxg.successors(nid))  # type: ignore
            return set(nxg.neighbors(nid))

        # obj connected to room
        try:
            # obj_ids = list(graph.get_neighbourhood_graph(room_id).filter_graph_by_node_types(["object"]).get_nodes_ids())
            obj_ids = [n for n in _neighbors_any_dir_local(room_id) if nxg.nodes[n].get("type") == "object"]
            nodes_to_remove.update(obj_ids)
        except Exception:
            pass
        
        # ws connected to room
        if remove_planes:
            ws_ids = []
            try:
                # ws_ids = list(graph.get_neighbourhood_graph(room_id).filter_graph_by_node_types(["ws"]).get_nodes_ids())
                ws_ids = [n for n in _neighbors_any_dir_local(room_id) if nxg.nodes[n].get("type") == "ws"]
                nodes_to_remove.update(ws_ids)
            except Exception:
                pass

            # wall + wall_ws connected to ws
            for ws_id in ws_ids:
                try:
                    # wall_ids = list(graph.get_neighbourhood_graph(room_id).filter_graph_by_node_types(["wall"]).get_nodes_ids())
                    wall_ids = [n for n in _neighbors_any_dir_local(ws_id) if nxg.nodes[n].get("type") == "wall"]
                    nodes_to_remove.update(wall_ids)
                except Exception:
                    pass

                # make sure wall_ws is also removed due to inconsistency 
                try:
                    # wall_ws_ids = list(graph.get_neighbourhood_graph(room_id).filter_graph_by_node_types(["wall_ws"]).get_nodes_ids())
                    wall_ws_ids = [n for n in _neighbors_any_dir_local(ws_id) if nxg.nodes[n].get("type") == "wall_ws"]
                    nodes_to_remove.update(wall_ws_ids)
                except Exception:
                    pass

        # remove only nodes that still exist
        existing = set(graph.get_nodes_ids())
        graph.remove_nodes([nid for nid in nodes_to_remove if nid in existing])
        return graph 
    
    def deconstruct_graph_room_by_room(
            self,
            graph: GraphWrapper,
            save_dir: Optional[str] = None,
            include_init: bool = True,
            return_sequence: bool = False,
            save_filename: str = "deconstruction_sequence.pkl",
    ):
        """
        Function to incrementally deconstruct a mature graph:
        - pick random building
        - top floor down to bottom floor
        - remove room one by one
        - when a floor has no rooms left -> delete floor node
        - when building has no floors left -> delete building node
        Saves and returns a sequence of GW snapshots. 
        
        Args:
            :param graph: graph to deconstruct
            :param save_dir: path to save incremental snapshots
            :param seed: seed for deconstruction using random function
            :param include_init: include initial passed graph in new save dir
            :param return_sequence: decide whether function returns in-memory sequence or not
        """
        debug_prints = False

        # inc_graph = copy.deepcopy(graph)
        nx_base = graph.graph
        nx_concrete = nx_base.copy()

        if not nx_concrete.is_directed():
            nx_concrete = nx_concrete.to_directed()

        if debug_prints:
            print("[DBG] input graph:", nx_concrete.number_of_nodes(), "nodes,", nx_concrete.number_of_edges(), "edges")

        inc_graph = GraphWrapper(graph_obj=nx_concrete)
        nxg = inc_graph.graph

        if self.seed:
            rng = random.Random(self.seed)
        else:
            rng = random.Random()

        # prep save location
        out_dir = None
        if save_dir is not None:
            out_dir = Path(save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        # init sequence pairs 
        seq_pair: list[tuple[GraphWrapper, GraphWrapper]] = []

        def _snapshot(gw: GraphWrapper) -> GraphWrapper:
            return copy.deepcopy(gw)

        # notify print
        print(f"Running incremental deconstruction: save_dir: {save_dir}")

        # include initial graph 
        if include_init:
            seq_pair.append((_snapshot(inc_graph), _snapshot(inc_graph)))

        # define planes to remove second
        PLANE_TYPES = {"ws", "wall", "wall_ws", "wallsurface"}

        def _z_of(nid):
            try:
                center = np.asarray(inc_graph.get_attributes_of_node(nid).get("center", [0.0,0.0,0.0]), dtype=float)
                if center.shape[0] == 2:
                    return 0.0
                return float(center[2])
            except Exception:
                return 0.0
            
        def _neighbors_any_dir(nid):
            if inc_graph.is_directed():
                return set(nxg.predecessors(nid)) | set(nxg.successors(nid))  # type: ignore
            return set(nxg.neighbors(nid))
            
        def _nodes_of_type(type: str):  # helper to type less :) 
            return [n for n, d in nxg.nodes(data=True) if d.get("type") == type]
        
        def _has_child_of_type(pid, child_type: str) -> bool:
            if pid not in set(inc_graph.get_nodes_ids()):
                return False
            return any(nxg.nodes[n].get("type") == child_type for n in _neighbors_any_dir(pid))
        
        def _remove_if_empty_floor_building_city(fid, bid):
            """
            remove higherlevel nodes
            -> remove floor if it has no more rooms,
            -> remove building if it has no more floors,
            -> remove city if it has no more builidngs,
            all in memeory; to avoid orphaned nodes in snapshot
            """
            # remove floor
            if fid is not None and fid in set(inc_graph.get_nodes_ids()):
                if not _has_child_of_type(fid, "room"):
                    inc_graph.remove_nodes([fid])

            # remove building 
            if bid is not None and bid in set(inc_graph.get_nodes_ids()):
                if not _has_child_of_type(bid, "floor"):
                    inc_graph.remove_nodes([bid])

            # remove city
            if not _nodes_of_type("building"):
                cid = _nodes_of_type("city")
                if cid:
                    inc_graph.remove_nodes(cid)
        
        def _purge_empty_hierarchy_nodes():
            """
            enforce no orphans, ensure atomic deconstruction
            - no floor wtihout rooms
            - no bulding without floor
            """
            while True:
                removed = False
                existing = set(inc_graph.get_nodes_ids())

                # remove orphan floors
                floor_ids = [n for n, d in nxg.nodes(data=True) if d.get("type") == "floor" and n in existing]
                orphan_floors = []
                for fid in floor_ids:
                    has_room = any(nxg.nodes[n].get("type") == "room" for n in _neighbors_any_dir(fid))
                    if not has_room:
                        orphan_floors.append(fid)
                if orphan_floors:
                    inc_graph.remove_nodes(orphan_floors)
                    removed = True
                    existing = set(inc_graph.get_nodes_ids())

                # remove orphan buildings
                building_ids = [n for n, d in nxg.nodes(data=True) if d.get("type") == "building" and n in existing]
                orphan_buildings = []
                for bid in building_ids:
                    has_room = any(nxg.nodes[n].get("type") == "floor" for n in _neighbors_any_dir(bid))
                    if not has_room:
                        orphan_buildings.append(bid)
                if orphan_buildings:
                    inc_graph.remove_nodes(orphan_buildings)
                    removed = True
                    existing = set(inc_graph.get_nodes_ids())

                # remove orphan city
                city_ids = [n for n, d in nxg.nodes(data=True) if d.get("type") == "city" and n in existing]
                if city_ids:
                    has_building = any(d.get("type") == "building" for n, d in nxg.nodes(data=True) if n in existing)

                    if not has_building:
                        inc_graph.remove_nodes(city_ids)
                        removed = True
                        existing = set(inc_graph.get_nodes_ids())

                if not removed:
                    break

        while True:
            # building_ids = list(inc_graph.filter_graph_by_node_types(["building"]).get_nodes_ids())  # incredilby slow
            building_ids = [n for n, d in nxg.nodes(data=True) if d.get("type") == "building"]
            if not building_ids:
                break

            # choose random building
            bid = rng.choice(building_ids)

            # handle directionality
            if not inc_graph.is_directed():
                neigh_nodes = set(nxg.neighbors(bid)) | {bid}
            else:
                neigh_nodes = set(nxg.predecessors(bid)) | set(nxg.successors(bid)) | {bid} # type: ignore

            floor_ids = [n for n in neigh_nodes if nxg.nodes[n].get("type") == "floor"]
            floor_ids = sorted(set(floor_ids), key=_z_of, reverse=True)  # reverse: top -> bottom

            # remove building if no floors connected to prevent spinning
            if not floor_ids:
                inc_graph.remove_nodes([bid])
                _purge_empty_hierarchy_nodes()
                continue

            if debug_prints:
                print(f"[DBG] picked building {bid} with {len(floor_ids)} floors (top z={_z_of(floor_ids[0]) if floor_ids else None})")

            for fid in floor_ids:
                if fid not in set(inc_graph.get_nodes_ids()):
                    continue

                # rooms on current floor 
                room_ids = [n for n in _neighbors_any_dir(fid) if nxg.nodes[n].get("type") == "room"]

                # remove floor if no rooms connect to preven endless loop
                if not room_ids:
                    _remove_if_empty_floor_building_city(fid=fid, bid=bid)
                    _purge_empty_hierarchy_nodes()
                    continue

                while room_ids:
                    rid = rng.choice(room_ids)

                    # remove room, ws, wall, obj
                    self._remove_room_subgraph(inc_graph, rid, remove_planes=False)

                    # atomic cascade removal 
                    _remove_if_empty_floor_building_city(fid=fid, bid=bid)
                    _purge_empty_hierarchy_nodes()

                    # recalculate hierarchy nodes XY positions
                    try:
                        inc_graph.recalculate_hierarchy_centers()
                    except Exception as e:
                        if debug_prints:
                            print(f"[DBG] recalc failed (A): {e}")

                    # save snapshot after removal
                    # _save_snapshot()
                    snap_A = _snapshot(inc_graph)

                    existing = set(inc_graph.get_nodes_ids())
                    plane_ids = [n for n, d in nxg.nodes(data=True) if d.get("type") in PLANE_TYPES and n in existing]

                    planes_to_remove = []
                    for pid in plane_ids:
                        has_room_neigh = any(nxg.nodes[n].get("type") == "room" for n in _neighbors_any_dir(pid))
                        if not has_room_neigh:
                            planes_to_remove.append(pid)

                    if planes_to_remove:
                        inc_graph.remove_nodes(planes_to_remove)

                    # atomic cascade
                    _remove_if_empty_floor_building_city(fid=fid, bid=bid)
                    _purge_empty_hierarchy_nodes()

                    # recalculate hierarchy nodes XY positions
                    try:
                        inc_graph.recalculate_hierarchy_centers()
                    except Exception as e:
                        if debug_prints:
                            print(f"[DBG] recalc failed (B): {e}")


                    snap_B = _snapshot(inc_graph)

                    seq_pair.append((snap_A, snap_B))

                    # refresh room list
                    if fid not in set(inc_graph.get_nodes_ids()):
                        break
                    room_ids = [n for n in _neighbors_any_dir(fid) if nxg.nodes[n].get("type") == "room"]

        if out_dir is not None:
            with open(out_dir / save_filename, "wb") as f:
                pickle.dump(seq_pair, f, protocol=pickle.HIGHEST_PROTOCOL)

        return seq_pair if return_sequence else inc_graph
    

    def include_observations(self, working_graph, pp_settings):
        graph_sequence = [copy.deepcopy(working_graph)]
        if "room" in pp_settings["elements"].keys():
            remaining_rooms_ids = list(working_graph.filter_graph_by_node_types("room").get_nodes_ids())
            n_rooms_to_remove = random.randint(pp_settings["elements"]["room"][0], pp_settings["elements"]["room"][1])
            
            while len(remaining_rooms_ids) > n_rooms_to_remove:
                rooms_to_remove = random.sample(remaining_rooms_ids, n_rooms_to_remove)
                for room_id in rooms_to_remove:
                    working_graph = self.dropout_by_hierarchy(room_id, working_graph, update_higher_nodes=True, remove_lower_nodes=True)
                graph_sequence.append(copy.deepcopy(working_graph))

                remaining_rooms_ids = list(working_graph.filter_graph_by_node_types("room").get_nodes_ids())
                n_rooms_to_remove = random.randint(pp_settings["elements"]["room"][0], pp_settings["elements"]["room"][1])

        return graph_sequence
    
    def compose_a_s_graphs(self, nxdataset):

        composed_datset = []

        for graphs_list in nxdataset:
            a_graph = copy.deepcopy(graphs_list[0]).upgrade_objects_type()

            extended_s_graphs = self.extend_nxdataset(graphs_list, "", "s_graphs")["train"]
            extended_s_graphs = [g[0] for g in extended_s_graphs]

            composed_datset.append((a_graph, extended_s_graphs))
        
        return composed_datset
            

    def extend_nxdataset(self, nxdataset, new_edge_type, stage):
        print(f"SyntheticDatasetGenerator: ", Fore.GREEN + "Extending Dataset" + Fore.WHITE)
        new_nxdataset = []

            
        def apply_postprocess(self, pp_settings, working_graph):

            if pp_settings["pp_name"] == "filter":
                working_graph = self.get_filtered_graph(working_graph, pp_settings["nodes"],pp_settings["edges"])

            if pp_settings["pp_name"] == "add_gt":
                possible_edge_types = copy.deepcopy(sorted(list(working_graph.get_all_edge_types())))
                for source_node_id, target_node_id, edge_attrs in copy.deepcopy(working_graph.get_attributes_of_all_edges()):
                    source_node_type = working_graph.get_attributes_of_node(source_node_id)["type"]
                    target_node_type = working_graph.get_attributes_of_node(target_node_id)["type"]
                    # if (source_node_type, target_node_type) in self.settings["initial_features"]["edges"]:
                    #     min_dist = [np.linalg.norm(working_graph.get_attributes_of_node(source_node_id)["center"] - working_graph.get_attributes_of_node(target_node_id)["center"])]
                        # rel_pos_1, centroids_distance, angle_centroid_degrees, angle_normals = relative_geometry(working_graph.get_attributes_of_node(source_node_id),working_graph.get_attributes_of_node(target_node_id))
                        # feature_dict = {"min_dist": min_dist, "relative_pos": rel_pos_1[:2], "centroids_distance": centroids_distance, "angle_centroid_degrees": angle_centroid_degrees, "relative_ang_normal": angle_normals}
                        # embedding_builder = NodeEdgeFeatureEmbeddingBuildier("edge", copy.deepcopy(feature_dict))
                        # x_straight = embedding_builder.build_embedding(self.settings["initial_features"]["edges"][tuple(["ws","ws"])])
                    
                        # rel_pos_1, centroids_distance, angle_centroid_degrees, angle_normals = relative_geometry(working_graph.get_attributes_of_node(target_node_id),working_graph.get_attributes_of_node(source_node_id))
                        # feature_dict = {"min_dist": min_dist, "relative_pos": rel_pos_1[:2], "centroids_distance": centroids_distance, "angle_centroid_degrees": angle_centroid_degrees, "relative_ang_normal": angle_normals}
                        # embedding_builder.update_feature_dictionary(feature_dict)
                        # x_inversed = embedding_builder.build_embedding(self.settings["initial_features"]["edges"][tuple(["ws","ws"])])
                    # else:
                    #     [x_straight, x_inversed] = [[],[]]
                    working_graph.update_edge_attrs((source_node_id, target_node_id), {"label":possible_edge_types.index(edge_attrs["type"])+1, "viz_feat" : 'green', "type" : new_edge_type, "linewidth":1.0, "alpha":0.5})
                    working_graph.add_edges([(target_node_id, source_node_id, {"label":possible_edge_types.index(edge_attrs["type"])+1, "viz_feat" : 'green', "type" : new_edge_type, "linewidth":1.0, "alpha":0.5})])
            
            elif pp_settings["pp_name"] == "add_x":
                for [node_id, node_attrs] in copy.deepcopy(working_graph.get_attributes_of_all_nodes()):
                    if node_attrs["type"] in self.settings["initial_features"]["nodes"]:
                        embedding_builder = NodeEdgeFeatureEmbeddingBuildier("node", node_attrs)
                        x = embedding_builder.build_embedding(self.settings["initial_features"]["nodes"][node_attrs["type"]])
                        working_graph.update_node_attrs(node_id, {"x":x})

                for (source_node_id, target_node_id, edge_attrs) in copy.deepcopy(working_graph.get_attributes_of_all_edges()):
                    source_node_type = working_graph.get_attributes_of_node(source_node_id)["type"]
                    target_node_type = working_graph.get_attributes_of_node(target_node_id)["type"]
                    if (source_node_type, target_node_type) in self.settings["initial_features"]["edges"]:
                        ### TODO Needs generalization this is hardcoded
                        min_dist = [np.linalg.norm(working_graph.get_attributes_of_node(source_node_id)["center"] - working_graph.get_attributes_of_node(target_node_id)["center"])]
                        rel_pos_1, centroids_distance, angle_centroid_degrees, angle_normals = relative_geometry(working_graph.get_attributes_of_node(source_node_id),working_graph.get_attributes_of_node(target_node_id))
                        feature_dict = {"min_dist": min_dist, "relative_pos": rel_pos_1[:2], "centroids_distance": centroids_distance, "angle_centroid_degrees": angle_centroid_degrees, "relative_ang_normal": angle_normals}
                        edge_attrs.update(feature_dict)
                        ### TODO End
                        embedding_builder = NodeEdgeFeatureEmbeddingBuildier("edge", copy.deepcopy(edge_attrs))
                        x = embedding_builder.build_embedding(self.settings["initial_features"]["edges"][tuple([source_node_type,target_node_type])])
                        working_graph.update_edge_attrs((source_node_id, target_node_id), {"x":x})



            elif pp_settings["pp_name"] == "remove_all_edges":
                    working_graph.remove_all_edges()

            elif pp_settings["pp_name"] == "dropout":
                ### room dropout
                if pp_settings["room"] > 0.:
                    room_node_ids = copy.deepcopy(list(working_graph.filter_graph_by_node_types(["room"]).get_nodes_ids()))
                    # node_ids_selected = []
                    for room_node_id in room_node_ids:
                        left_rooms = list(working_graph.filter_graph_by_node_types(["room"]).get_nodes_ids())
                        if len(left_rooms) > 1 and np.random.random_sample() < pp_settings["room"]:
                            working_graph = self.dropout_by_hierarchy(room_node_id, working_graph, update_higher_nodes=True, remove_lower_nodes = pp_settings["remove_lower"])

                ### wall dropout
                if pp_settings["wall"] > 0.:
                    room_node_ids = copy.deepcopy(list(working_graph.filter_graph_by_node_types(["wall"]).get_nodes_ids()))
                    # node_ids_selected = []
                    for room_node_id in room_node_ids:
                        left_rooms = list(working_graph.filter_graph_by_node_types(["wall"]).get_nodes_ids())
                        if len(left_rooms) > 1 and np.random.random_sample() < pp_settings["wall"]:
                            working_graph = self.dropout_by_hierarchy(room_node_id, working_graph, update_higher_nodes=False, remove_lower_nodes = pp_settings["remove_lower"])

                ### ws dropout  TODO FIX
                if pp_settings["ws"] > 0.:
                    ws_node_ids = list(working_graph.filter_graph_by_node_types(["ws"]).get_nodes_ids())
                    # node_ids_selected = []
                    for ws_node_id in ws_node_ids:
                        # visualize_nxgraph(working_graph.get_neighbourhood_graph(ws_node_id), "test", visualize_alone=True)
                        # for e in working_graph.get_neighbourhood_graph(ws_node_id).get_attributes_of_all_edges():
                        #     print(f"dbg e[2][type] {e[2]['type']}")
                        # visualize_nxgraph(working_graph.get_neighbourhood_graph(ws_node_id).filter_graph_by_edge_types(["ws_same_room"]).filterout_unparented_nodes(), "test 2", visualize_alone=True)
                        room_id = list(working_graph.get_neighbourhood_graph(ws_node_id).filter_graph_by_node_types(["room"]).get_nodes_ids())[0]
                        same_room_ws_node_ids = room_id = list(working_graph.get_neighbourhood_graph(room_id).filter_graph_by_node_types(["ws"]).get_nodes_ids())
                        # same_room_ws_node_ids = list(working_graph.get_neighbourhood_graph(ws_node_id).filter_graph_by_node_types(["ws"]).filterout_unparented_nodes().get_nodes_ids())
                        print(f"dbg same_room_ws_node_ids {same_room_ws_node_ids}")
                        # left_in_same_room_ws_node_ids = list(set(same_room_ws_node_ids) - set(node_ids_selected))
                        if len(same_room_ws_node_ids) > 2 and np.random.random_sample() < pp_settings["ws"]:
                            # node_ids_selected.append(ws_node_id)
                            working_graph = self.dropout_by_hierarchy(ws_node_id, working_graph, update_higher_nodes=True, remove_lower_nodes = pp_settings["remove_lower"])
                    # working_graph.remove_nodes(node_ids_selected)

            ### Include K nearest neighbouors edges
            elif pp_settings["pp_name"] == "K_near_neigh":
                node_ids = list(working_graph.filter_graph_by_node_types(pp_settings["types"]).get_nodes_ids())
                centers = np.array([working_graph.get_attributes_of_node(node_id)["center"] for node_id in node_ids])
                try:
                    kdt = KDTree(centers, leaf_size=30, metric='euclidean')
                except:
                    visualize_nxgraph(working_graph, "trial", visualize_alone=True)
                k = len(centers) if len(centers) <= pp_settings["max"]+1 else pp_settings["max"]+1
                query = kdt.query(centers, k=k, return_distance=False)
                query = np.array(list((map(lambda e: list(map(node_ids.__getitem__, e)), query))))
                base_nodes_ids = query[:, 0]
                all_target_nodes_ids = query[:, 1:]
                new_edges = []
                positive_gt_edge_ids = list(working_graph.get_edges_ids())
                
                for i, base_node_id in enumerate(base_nodes_ids):
                    target_nodes_ids = all_target_nodes_ids[i]
                    for target_node_id in target_nodes_ids:
                        base_node_type = working_graph.get_attributes_of_node(base_node_id)["type"]
                        target_node_type = working_graph.get_attributes_of_node(target_node_id)["type"]
                        tuple_direct, tuple_inverse = (base_node_id, target_node_id), (target_node_id, base_node_id)
                        if (base_node_type, target_node_type) in self.settings["initial_features"]["edges"]:
                            distance = [np.linalg.norm(working_graph.get_attributes_of_node(base_node_id)["center"] - working_graph.get_attributes_of_node(target_node_id)["center"])]
                            rel_pos_1, centroids_distance, angle_centroid_degrees, angle_normals = relative_geometry(working_graph.get_attributes_of_node(base_node_id),working_graph.get_attributes_of_node(target_node_id))
                            feature_dict = {"min_dist": distance, "relative_pos": rel_pos_1[:2], "centroids_distance": centroids_distance, "angle_centroid_degrees": angle_centroid_degrees, "relative_ang_normal": angle_normals}
                            embedding_builder = NodeEdgeFeatureEmbeddingBuildier("edge", copy.deepcopy(feature_dict))
                            x_straight = embedding_builder.build_embedding(self.settings["initial_features"]["edges"][tuple(["ws","ws"])])
                            
                            rel_pos_1, centroids_distance, angle_centroid_degrees, angle_normals = relative_geometry(working_graph.get_attributes_of_node(target_node_id),working_graph.get_attributes_of_node(base_node_id))
                            feature_dict = {"min_dist": distance, "relative_pos": rel_pos_1[:2], "centroids_distance": centroids_distance, "angle_centroid_degrees": angle_centroid_degrees, "relative_ang_normal": angle_normals}
                            embedding_builder.update_feature_dictionary(feature_dict)
                            x_inversed = embedding_builder.build_embedding(self.settings["initial_features"]["edges"][tuple(["ws","ws"])])
                        
                        else:
                            [x_straight, x_inversed] = [[],[]]

                        if tuple_direct in positive_gt_edge_ids or tuple_inverse in positive_gt_edge_ids: ### TODO merge use_gt with false
                            if not pp_settings["use_gt"]:
                                # TODO what to do then with the label. it does not matter?
                                new_edges.append((base_node_id, target_node_id,{"type": new_edge_type, "label": 1, "x":x_straight, "viz_feat" : 'g', "linewidth":1.0, "alpha":0.5}))
                                new_edges.append((target_node_id, base_node_id,{"type": new_edge_type, "label": 1, "x":x_inversed, "viz_feat" : 'g', "linewidth":1.0, "alpha":0.5}))
                                # new_edges.append((target_node_id, base_node_id,{"type": new_edge_type, "label": 1, "x":x_2, "viz_feat" : 'g', "linewidth":1.0, "alpha":0.5}))
                            # else:
                            #     new_edges.append((target_node_id, base_node_id,{"type": new_edge_type, "label": 0, "x":x, "viz_feat" : 'r', "linewidth":1.0, "alpha":0.5}))
                        else:
                            new_edges.append((base_node_id, target_node_id,{"type": new_edge_type, "label": 0, "x":x_straight, "viz_feat" : 'r', "linewidth":1.0, "alpha":0.5}))
                            new_edges.append((target_node_id, base_node_id,{"type": new_edge_type, "label": 0, "x":x_inversed, "viz_feat" : 'r', "linewidth":1.0, "alpha":0.5}))
                            # new_edges.append((target_node_id, base_node_id,{"type": new_edge_type, "label": 0, "x":x_2, "viz_feat" : 'r', "linewidth":1.0, "alpha":0.5}))
                working_graph.unfreeze()
                working_graph.add_edges(new_edges)

            ### Include random edges
            #{"pp_name": "K_rand_neigh", "max": 0, "types":["ws"], "use_gt":true}
            elif pp_settings["pp_name"] == "K_rand_neigh":
                nodes_ids = list(working_graph.filter_graph_by_node_types(pp_settings["types"]).get_nodes_ids())
                for base_node_id in nodes_ids:
                    potential_nodes_ids = copy.deepcopy(nodes_ids)
                    potential_nodes_ids.remove(base_node_id)
                    random.shuffle(potential_nodes_ids)
                    random_nodes_ids = potential_nodes_ids[pp_settings["max"]]

                    new_edges = []
                    for target_node_id in random_nodes_ids:
                        tuple_direct = (base_node_id, target_node_id)
                        tuple_inverse = (tuple_direct[1], tuple_direct[0])
                        if tuple_direct not in list(working_graph.get_edges_ids()) and tuple_inverse not in list(working_graph.get_edges_ids()):
                            ### TODO Include X
                            new_edges.append((tuple_direct[0], tuple_direct[1],{"type": new_edge_type, "label": 0, "viz_feat" : 'blue', "linewidth":1.0, "alpha":0.5}))

                working_graph.unfreeze()
                working_graph.add_edges(new_edges)

            elif pp_settings["pp_name"] == "ws_partial_occlusion":
                if pp_settings["ratio"] > 0.:
                    ws_node_ids = list(working_graph.filter_graph_by_node_types(["ws"]).get_nodes_ids())
                    for ws_node_id in ws_node_ids:
                        if np.random.random_sample() < pp_settings["ratio"]:
                            ws_attrs = working_graph.get_attributes_of_node(ws_node_id)
                            center = ws_attrs["center"]
                            length = ws_attrs["length"]
                            normal = ws_attrs["normal"]
                            rotation = R.from_euler('z', -90, degrees=True)
                            ws_direction = rotation.apply(copy.deepcopy(normal))
                            ws_direction /= np.linalg.norm(ws_direction)

                            if length> 1.:
                                new_length = length*random.uniform(0.5/length, (length-0.5)/length)
                                center_move_range = length - new_length
                                center_move = center_move_range*(np.random.random_sample() - 0.5)
                                new_center = center + ws_direction*center_move
                                new_limits = [new_center + ws_direction*new_length/2, new_center - ws_direction*new_length/2]
                                ws_attrs["center"] = new_center
                                ws_attrs["limits"] = new_limits
                                ws_attrs["viz"]["limits"] = new_limits
                                ws_attrs["length"] = new_length
            
            elif pp_settings["pp_name"] == "ws_split":
                if pp_settings["ratio"] > 0.:
                    ws_node_ids = list(working_graph.filter_graph_by_node_types(["ws"]).get_nodes_ids())
                    for ws_node_id in ws_node_ids:
                        if np.random.random_sample() < pp_settings["ratio"]:
                            ws_attrs = working_graph.get_attributes_of_node(ws_node_id)
                            center = ws_attrs["center"]
                            length = ws_attrs["length"]
                            normal = ws_attrs["normal"]
                            limits = ws_attrs["limits"]
                            rotation = R.from_euler('z', 90, degrees=True)
                            ws_direction = rotation.apply(normal)

                            n_splits = np.random.randint(1,4)
                            internal_split_lengths = np.sort([np.random.random_sample()*length for _ in range(n_splits)])

                            full_internal_split_lengths = np.concatenate([np.array([0.]), internal_split_lengths, np.array([length])])
                            full_internal_lengths = np.array([full_internal_split_lengths[i+1] - full_internal_split_lengths[i] for i in range(len(full_internal_split_lengths) - 1)])
                            min_length_mask = full_internal_lengths > 0.5
                            min_length_mask[-2] = min_length_mask[-2]*min_length_mask[-1]
                            internal_split_lengths_masked = internal_split_lengths[min_length_mask[:-1]]

                            init_limit = center - ws_direction*length/2

                            split_limits = [limits[0]]
                            for i in range(len(internal_split_lengths_masked)):
                                split_limits.append(init_limit + ws_direction*internal_split_lengths_masked[i])
                            split_limits.append(limits[1])
                            
                            neigh_room_IDs = working_graph.get_neighbourhood_graph(ws_node_id).filter_graph_by_node_types("room").get_nodes_ids()
                            neigh_wall_IDs = working_graph.get_neighbourhood_graph(ws_node_id).filter_graph_by_node_types("wall").get_nodes_ids()
                            ws_same_room_IDs = list(working_graph.get_neighbourhood_graph(ws_node_id).filter_graph_by_edge_types("ws_same_room").filterout_unparented_nodes().get_nodes_ids())
                            ws_same_wall_IDs = working_graph.get_neighbourhood_graph(ws_node_id).filter_graph_by_edge_types("ws_same_wall").filterout_unparented_nodes().get_nodes_ids()
                            nodes_to_add = []
                            edges_to_add = []
                            
                            new_node_IDs = []
                            
                            for i in range(len(internal_split_lengths_masked) + 1):
                                new_limits = [split_limits[i], split_limits[i+1]]

                                new_node_ID = max(working_graph.get_nodes_ids()) + i + 1
                                new_node_IDs.append(new_node_ID)

                                new_ws_attrs = copy.deepcopy(ws_attrs)
                                new_ws_attrs["center"] = (new_limits[0] + new_limits[1]) / 2
                                new_ws_attrs["limits"] = new_limits
                                new_ws_attrs["viz"]["limits"] = new_limits
                                new_ws_attrs["length"] = np.linalg.norm(new_limits[1] - new_limits[0])

                                nodes_to_add.append((new_node_ID,new_ws_attrs))

                                for neigh_room_ID in neigh_room_IDs:
                                    if neigh_room_ID != ws_node_id:
                                        edges_to_add.append((neigh_room_ID, new_node_ID, {"type": "ws_belongs_room", "viz_feat" : 'red', "linewidth":1.0, "alpha":0.5}))
                                        edges_to_add.append((new_node_ID, neigh_room_ID, {"type": "ws_belongs_room", "viz_feat" : 'red', "linewidth":1.0, "alpha":0.5}))
                                
                                for neigh_wall_ID in neigh_wall_IDs:
                                    if neigh_wall_ID != ws_node_id:
                                        edges_to_add.append((neigh_wall_ID, new_node_ID, {"type": "ws_belongs_wall", "viz_feat": "m", "linewidth":1.0, "alpha":0.5}))
                                        edges_to_add.append((new_node_ID, neigh_wall_ID, {"type": "ws_belongs_wall", "viz_feat": "m", "linewidth":1.0, "alpha":0.5}))

                                for ws_same_room_ID in ws_same_room_IDs + new_node_IDs[:-1]:
                                    if ws_same_room_ID != ws_node_id:
                                        edges_to_add.append((ws_same_room_ID, new_node_ID, {"type": "ws_same_room", "viz_feat": "b", "linewidth":1.0, "alpha":0.5}))
                                        edges_to_add.append((new_node_ID, ws_same_room_ID, {"type": "ws_same_room", "viz_feat": "b", "linewidth":1.0, "alpha":0.5}))
                                
                                for ws_same_wall_ID in ws_same_wall_IDs:
                                    if ws_same_wall_ID != ws_node_id:
                                        edges_to_add.append((ws_same_wall_ID, new_node_ID, {"type": "ws_same_wall", "viz_feat": "orange", "linewidth":1.0, "alpha":0.5}))
                                        edges_to_add.append((new_node_ID, ws_same_wall_ID, {"type": "ws_same_wall", "viz_feat": "orange", "linewidth":1.0, "alpha":0.5}))

                            working_graph.add_nodes(nodes_to_add)
                            working_graph.add_edges(edges_to_add)
                            working_graph.remove_nodes([ws_node_id])

            elif pp_settings["pp_name"] == "merge_room":
                working_graph = self.merge_rooms(pp_settings, working_graph)
            elif pp_settings["pp_name"] == "plot":
                if pp_settings["msd"]: 
                    pl.plot_a_graph([working_graph.graph],viz_room_normals=True,viz_walls=False)
                else:
                    fig = visualize_nxgraph_3d(working_graph, pp_settings["fig_name"], visualize_alone=pp_settings["visualize_alone"])
                    # plt.show(block=True)
                if pp_settings["save_path"]:
                    fig.savefig(pp_settings["save_path"], bbox_inches='tight')
            elif pp_settings["pp_name"] == "remove_self_loops":
                working_graph.remove_self_loops()
            elif pp_settings["pp_name"] == "relabel_nodes":
                working_graph.relabel_nodes(mapping = pp_settings["mapping"], copy=pp_settings["copy"])
            elif pp_settings["pp_name"] == "unfreeze":
                working_graph.unfreeze()
            elif pp_settings["pp_name"] == "add_global_noise":
                working_graph = self.apply_global_noise(working_graph, pp_settings)

            elif pp_settings["pp_name"] == "add_local_noise":
                emergent_concepts = ["wall", "room", "floor", "building", "city"]
                for node_id, node_attrs in working_graph.get_attributes_of_all_nodes():
                    if "center" in node_attrs and node_attrs.get("type") == "ws" and "ws" in pp_settings["entities"]:
                        # Calculate local translation (only for the center)
                        local_translation = np.array(pp_settings["translation"]) * (np.random.rand(2) - 0.5)

                        # Calculate local rotation (for normal and limits)
                        local_rotation_angle = (np.random.rand() - 0.5) * 360 * pp_settings["rotation"]
                        rotation_matrix_2d = R.from_euler("Z", local_rotation_angle, degrees=True).as_matrix()[:2, :2]
                        rotation_matrix_3d = R.from_euler("Z", local_rotation_angle, degrees=True).as_matrix()[:3, :3]

                        # --- TRANSLATE ONLY THE CENTER ---
                        new_center = node_attrs["center"][:2] + local_translation
                        node_attrs["center"][0] = new_center[0]
                        node_attrs["center"][1] = new_center[1]
                        node_attrs["viz"]["center"][0] = new_center[0]
                        node_attrs["viz"]["center"][1] = new_center[1]

                        # --- ROTATE ONLY THE NORMAL ---
                        if "normal" in node_attrs:
                            new_normal = rotation_matrix_3d @ node_attrs["normal"]
                            node_attrs["normal"][0] = new_normal[0]
                            node_attrs["normal"][1] = new_normal[1]

                        # --- ROTATE THE LIMITS (without translation) ---
                        if "limits" in node_attrs:
                            center = node_attrs["center"]  # updated center
                            rotated_limits = []
                            for point in node_attrs["limits"]:
                                vec = np.array(point) - center  # vector relative to the center
                                rotated_point = center + rotation_matrix_3d @ vec  # rotate around the center
                                rotated_limits.append(rotated_point.tolist())
                            node_attrs["limits"] = rotated_limits


                    elif "center" in node_attrs and node_attrs.get("type") in emergent_concepts and node_attrs.get("type") in pp_settings["entities"]:
                        # Calculate local translation (only for the center)
                        local_translation = np.array(pp_settings["translation"]) * (np.random.rand(2) - 0.5)

                        new_center = node_attrs["center"][:2] + local_translation
                        node_attrs["center"][0] = new_center[0]
                        node_attrs["center"][1] = new_center[1]
                        node_attrs["viz"]["center"][0] = new_center[0]
                        node_attrs["viz"]["center"][1] = new_center[1]

                    working_graph.update_node_attrs(node_id, node_attrs)

            elif pp_settings["pp_name"] == "msd_adaptation":
                nodes_to_remove = []
                for node_id, node_attrs in working_graph.get_attributes_of_all_nodes():
                    #('1588_3c3b1d6ca8b4b9092480b8c75f9eaa81_wall_6_0',
                    #  {'geom': [array([ 2.99975914, 10.91433429]), array([-2.85569845,  3.93607167])],
                    #  'polygon': [(2.9997591397090204, 10.914334285729849), (-2.855698447473351, 3.9360716699222333), (-2.855698447473351, 3.946071669922233), (2.9997591397090204, 10.924334285729849), (2.9997591397090204, 10.914334285729849)],
                    #  'center': [0.07203034611783465, 7.425202977826041, 1.3], 'normal': array([ 0.76604444, -0.64278761,  0.        ]),
                    #  'width': 9.109474885550195,
                    #  'type': 'wall_ws',
                    #  'category': 9})
                    if node_attrs["type"] in ["door_ws", "window_ws", "wall_ws", "door", "window", "wall"]:
                        nodes_to_remove.append(node_id)
                    if "center" in node_attrs :
                        # remove z coordinate from center and normal
                        node_attrs["center"] = node_attrs["center"][:2]
                        node_attrs["normal"] = node_attrs["normal"][:2]
                        node_attrs["viz"]["center"] = node_attrs["center"][:2]
                        #change field name from width to length
                        if("width" in node_attrs):
                            node_attrs["length"] = node_attrs.pop("width")
                    if "ws" in node_attrs["type"]:
                        length = node_attrs["length"]
                        center = np.array(node_attrs["center"][:2])
                        normal = np.array(node_attrs["normal"][:2])

                        # Rotazioni di ±90 gradi
                        normal_pos_90 = np.array([-normal[1], normal[0]])   # +90°
                        normal_neg_90 = np.array([normal[1], -normal[0]])   # -90°

                        # Calcolo dei limiti
                        half_length = length / 2.0
                        limit_1 = (center + half_length * normal_pos_90).tolist()
                        limit_2 = (center + half_length * normal_neg_90).tolist()

                        node_attrs["limits"] = [limit_1, limit_2]

                        # not doing in this way cause I can better check info passed to the network
                        # node_attrs["limits"] = [node_attrs["geom"][0], node_attrs["geom"][1]]
                working_graph.remove_nodes(nodes_to_remove)

            elif pp_settings["pp_name"] == "add_floor_node":
                working_graph = self.add_floor_node(working_graph)

            elif pp_settings["pp_name"] == "add_stories":
                working_graph = self.add_stories(working_graph, pp_settings["n_stories"], pp_settings["add_floor_nodes"])

            elif pp_settings["pp_name"] == "add_buildings":
                working_graph = self.add_buildings(working_graph, pp_settings["n_buildings"], pp_settings["area_shape"], pp_settings["area_radius"])

            elif pp_settings["pp_name"] == "add_random_objects":
                working_graph = self.add_random_objects(working_graph, pp_settings["max"], pp_settings["distrib"])

            elif pp_settings["pp_name"] == "merge_edge_types":
                working_graph = self.merge_edge_types(working_graph, pp_settings["common_type"])

            elif pp_settings["pp_name"] == "to_undirected":
                working_graph.to_undirected()

            elif pp_settings["pp_name"] == "incremental_observations":
                working_graph = self.include_observations(working_graph, pp_settings)

            elif pp_settings["pp_name"] == "update_viz":
               working_graph._add_complete_viz_attributes_to_graph(self.viz_center_offsets, self.node_viz_feat_mapping)

            elif pp_settings["pp_name"] == "recalculate_positions":
                working_graph.recalculate_hierarchy_centers()
            
            elif pp_settings["pp_name"] == "incremental_deconstruct":
                # get config settings
                # print(f"[DBG] apply_postprocess: 'incremental_deconstruct'")
                return_sequence = bool(pp_settings.get("return_sequence", False))
                include_init = bool(pp_settings.get("include_init", True))
                save_dir = pp_settings.get("save_dir", None)
                seed = pp_settings.get("seed", None)

                working_graph = self.deconstruct_graph_room_by_room(
                    working_graph,
                    save_dir=save_dir,
                    seed=seed,
                    include_init=include_init,
                    return_sequence=return_sequence,
                )

            elif pp_settings["pp_name"] == "save_snapshot":
                suffix = pp_settings.get("suffix", "processed")

                # use timestamp to avoid overwriting
                ts = int(time.time() * 1000)
                file_name = f"graph_{suffix}_{ts}.pkl"
                file_path = self.save_dir / file_name

                self.save_wrappers_to_pickle([working_graph], str(file_path))

                if self.logger:
                    self.logger.info(f"Saved snapshot: {file_name}")

            elif pp_settings["pp_name"] == "randomize_edges":
                working_graph = working_graph.randomize_edges(pp_settings["percentage"])

            elif pp_settings["pp_name"] == "upgrade_objects_type":
                working_graph = working_graph.upgrade_objects_type()

            return working_graph

        for i in tqdm.tqdm(range(len(nxdataset)), colour="green"):
            nxdata = nxdataset[i]
            base_graph = copy.deepcopy(nxdata)
            pp_settings_list = self.settings["postprocess"][stage]
            for pp_settings in pp_settings_list:
                # part_1_end = time.time()
                base_graph = apply_postprocess(self, pp_settings, base_graph)
                # part_2_end = time.time()
                # print(f"dbg elapsed time in pp {pp_settings['pp_name']}: {part_2_end - part_1_end}")

            def _as_sequence(graph):
                return graph if isinstance(graph, list) else [graph]
            
            sequence = _as_sequence(base_graph)
            valid_sequence = []

            for graph in sequence:
                if isinstance(graph, GraphWrapper):
                    if len(graph.get_nodes_ids()) > 0 and len(graph.get_edges_ids()) > 0:
                        valid_sequence.append(graph)

            if valid_sequence:
                new_nxdataset.append(valid_sequence)

            # legacy code commented out 
            #if type(base_graph) == GraphWrapper and len(base_graph.get_nodes_ids()) > 0 and len(base_graph.get_edges_ids()) > 0:
            #    new_nxdataset.append(base_graph)
            #elif type(base_graph) == list and base_graph:
            #    new_nxdataset.extend([base_graph])

        val_start_index = int(len(nxdataset)*(1-self.settings["training_split"]["val"]-self.settings["training_split"]["test"]))
        test_start_index = int(len(nxdataset)*(1-self.settings["training_split"]["test"]))
        extended_nxdatset = {"train" : new_nxdataset[:val_start_index], "val" : new_nxdataset[val_start_index:test_start_index],"test" : new_nxdataset[test_start_index:]}
        self.graphs["extended"] = new_nxdataset
        
        return extended_nxdatset
    

    def merge_rooms(self, pp_settings, working_graph):
        wall_nodes_ids = copy.deepcopy(working_graph).filter_graph_by_node_types("wall").get_nodes_ids()
        for wall_node_id in wall_nodes_ids:
            if np.random.random_sample() < pp_settings["ratio"]:
                node_ids_to_remove = []
                ws_nodes_ids = copy.deepcopy(working_graph).get_neighbourhood_graph(wall_node_id).filter_graph_by_node_types("ws").get_nodes_ids()
                room_nodes_ids = []
                rooms_ws_nodes_ids = []
                num_related_walls = []
                for ws_node_id in ws_nodes_ids:
                    room_nodes_ids.append(list(copy.deepcopy(working_graph).get_neighbourhood_graph(ws_node_id).filter_graph_by_node_types("room").get_nodes_ids())[0])
                    rooms_ws_nodes_ids.append(list(copy.deepcopy(working_graph).get_neighbourhood_graph(room_nodes_ids[-1]).filter_graph_by_node_types("ws").get_nodes_ids()))
                    num_related_walls.append(len(list(copy.deepcopy(working_graph).get_neighbourhood_graph(ws_node_id).filter_graph_by_node_types("wall").get_nodes_ids())))
                num_related_ws = [len(i) for i in rooms_ws_nodes_ids]
                elegibility_condition = num_related_walls.count(1) == 1 and (num_related_ws[0] < 5 and num_related_ws[1] < 5)
                if elegibility_condition:
                    for i, ws_node_id in enumerate(ws_nodes_ids):
                        related_walls = list(copy.deepcopy(working_graph).get_neighbourhood_graph(ws_node_id).filter_graph_by_node_types("wall").get_nodes_ids())
                        if len(related_walls) == 1:
                            node_ids_to_remove.append(ws_node_id)
                        else:

                            ### shorten the ws
                            ws_node_attrs = working_graph.get_attributes_of_node(ws_node_id)
                            other_room_ws_centers = [working_graph.get_attributes_of_node(node_id)["center"] for node_id in rooms_ws_nodes_ids[1-i]]
                            other_room_ws_closest_points = [closest_point_on_segment(center, ws_node_attrs["limits"][0], ws_node_attrs["limits"][1]) for center in other_room_ws_centers]
                            distances = [[distance_between_points(point, ws_node_attrs["limits"][0]),distance_between_points(point, ws_node_attrs["limits"][1])] for point in other_room_ws_closest_points]
                            distances = np.array(distances)
                            _, min_col = np.unravel_index(np.argmin(distances), distances.shape)
                            min_dist_idx = np.argmin(distances[:,1-min_col])
                            ws_node_attrs["limits"][min_col] = copy.deepcopy(np.array(other_room_ws_closest_points[min_dist_idx]))
                            ws_node_attrs["viz"]["limits"][min_col] = copy.deepcopy(np.array(other_room_ws_closest_points[min_dist_idx]))
                            ws_node_attrs["center"] = copy.deepcopy((np.array(ws_node_attrs["limits"][0]) + np.array(ws_node_attrs["limits"][1])) / 2)
                            ws_length = np.linalg.norm(ws_node_attrs["limits"][0] - ws_node_attrs["limits"][1])
                            # feature_dict = {"ws_center": ws_node_attrs["center"], "ws_normal": ws_node_attrs["normal"], "ws_length": ws_length}
                            # embedding_builder = NodeEdgeFeatureEmbeddingBuildier("node", feature_dict)
                            # ws_node_attrs["x"] = embedding_builder.build_embedding(self.settings["initial_features"]["nodes"]["ws"])
                            ws_node_attrs["length"] = ws_length
                            ### update shortened ws' wall's center
                            related_walls.remove(wall_node_id)
                            for related_wall in related_walls:
                                neigh_wall_ws = list(copy.deepcopy(working_graph).get_neighbourhood_graph(related_wall).filter_graph_by_node_types("ws").get_nodes_ids())
                                neigh_wall_ws.remove(ws_node_id)
                                new_wall_center = np.array((np.array(working_graph.get_attributes_of_node(neigh_wall_ws[0])["center"]) + np.array(ws_node_attrs["center"])) / 2)
                                wall_attrs = working_graph.get_attributes_of_node(related_wall)
                                wall_attrs["center"] = list(new_wall_center)
                                wall_attrs["viz"]['center'] = new_wall_center + self.viz_center_offsets["wall"]
                                # wall_attrs["x"] = new_wall_center

                    ### merge same plane ws
                    random.shuffle(room_nodes_ids)
                    room1_ws_nodes_ids = list(copy.deepcopy(working_graph).get_neighbourhood_graph(room_nodes_ids[0]).filter_graph_by_node_types("ws").get_nodes_ids())
                    room2_ws_nodes_ids = list(copy.deepcopy(working_graph).get_neighbourhood_graph(room_nodes_ids[1]).filter_graph_by_node_types("ws").get_nodes_ids())
                    combinations = list(itertools.product(room1_ws_nodes_ids, room2_ws_nodes_ids))
                    collinearity = [are_segments_collinear(working_graph.get_attributes_of_node(combination[0])["limits"], working_graph.get_attributes_of_node(combination[1])["limits"]) for combination in combinations]
                    collinearity_indeces = [index for index, value in enumerate(collinearity) if value]
                    for collinearity_index in collinearity_indeces:
                        if not len(set(combinations[collinearity_index]).intersection(set(node_ids_to_remove))) != 0:
                            combination = combinations[collinearity_index]
                            ws0_attrs_limits = working_graph.get_attributes_of_node(combination[0])["limits"]
                            ws1_attrs_limits = working_graph.get_attributes_of_node(combination[1])["limits"]
                            candidates_limits = list(itertools.product(ws0_attrs_limits, ws1_attrs_limits))
                            distances = [distance_between_points(points[0], points[1]) for points in candidates_limits]
                            # if min(distances) < wall_thickness*2:
                            if True:
                                new_limits = candidates_limits[np.argmax(distances)]
                                new_center = (new_limits[0] + new_limits[1]) / 2

                                ws0_attrs = working_graph.get_attributes_of_node(combination[0])
                                ws0_attrs["limits"] = list(new_limits)
                                ws0_attrs["viz"]["limits"] = list(new_limits)
                                ws0_attrs["center"] = new_center
                                # ws0_length = np.linalg.norm(ws0_attrs["limits"][0] - ws0_attrs["limits"][1])

                                # feature_dict = {"ws_center": ws0_attrs["center"][:2], "ws_normal": ws0_attrs["normal"][:2], "ws_length": ws0_length}
                                # embedding_builder = NodeEdgeFeatureEmbeddingBuildier("node", feature_dict)
                                # ws0_attrs["x"] = embedding_builder.build_embedding(self.settings["initial_features"]["nodes"]["ws"])

                                node_ids_to_remove.append(combination[1])
                                ws0_walls_ids = list(copy.deepcopy(working_graph).get_neighbourhood_graph(combination[0]).filter_graph_by_node_types("wall").get_nodes_ids())
                                ws1_walls_ids = list(copy.deepcopy(working_graph).get_neighbourhood_graph(combination[1]).filter_graph_by_node_types("wall").get_nodes_ids())
                                for ws1_wall_id in ws1_walls_ids:
                                    working_graph.add_edges([(combination[0], ws1_wall_id, {"type": "ws_belongs_wall", "x": [], "viz_feat": "m", "linewidth":1.0, "alpha":0.5})])
                                    neigh_wall_ws = list(copy.deepcopy(working_graph).get_neighbourhood_graph(ws1_wall_id).filter_graph_by_node_types("ws").get_nodes_ids())
                                    neigh_wall_ws.remove(combination[1])
                                    if combination[0] != neigh_wall_ws[0]:
                                        working_graph.add_edges([(combination[0], neigh_wall_ws[0], {"type": "ws_same_wall", "x": [], "viz_feat": "orange", "linewidth":1.0, "alpha":0.5})])

                                ### update merged ws' wall's center
                                for related_wall in ws0_walls_ids + ws1_walls_ids:
                                    neigh_wall_ws = list(copy.deepcopy(working_graph).get_neighbourhood_graph(related_wall).filter_graph_by_node_types("ws").get_nodes_ids())
                                    if combination[0] in neigh_wall_ws: neigh_wall_ws.remove(combination[0])
                                    if combination[1] in neigh_wall_ws: neigh_wall_ws.remove(combination[1])

                                    if neigh_wall_ws:
                                        new_wall_center = (np.array(working_graph.get_attributes_of_node(neigh_wall_ws[0])["center"]) + np.array(new_center)) / 2
                                        wall_attrs = working_graph.get_attributes_of_node(related_wall)
                                        wall_attrs["center"] = list(new_wall_center)
                                        wall_attrs["viz"]["center"] = new_wall_center + self.viz_center_offsets["wall"]
                                        wall_attrs["x"] = new_wall_center
                

                    ### update room centers
                    room1_ws_nodes_ids = list(copy.deepcopy(working_graph).get_neighbourhood_graph(room_nodes_ids[0]).filter_graph_by_node_types("ws").get_nodes_ids())
                    room2_ws_nodes_ids = list(copy.deepcopy(working_graph).get_neighbourhood_graph(room_nodes_ids[1]).filter_graph_by_node_types("ws").get_nodes_ids())
                    node_ids_to_remove.append(room_nodes_ids[1])
                    node_ids_to_remove.append(wall_node_id)
                    for node_id in room2_ws_nodes_ids:
                        attrs = working_graph.get_attributes_of_edge((node_id, room_nodes_ids[1]))
                        working_graph.add_edges([(node_id, room_nodes_ids[0], attrs)])
                    
                    ws_centers = [[working_graph.get_attributes_of_node(node_id)["center"]] for node_id in room1_ws_nodes_ids + room2_ws_nodes_ids]
                    ws_centers = np.concatenate(ws_centers, axis=0)
                    room_center = np.mean(ws_centers, axis=0)
                    room1_attrs = working_graph.get_attributes_of_node(room_nodes_ids[0])

                    room1_attrs["center"] = room_center
                    room1_attrs["x"] = room_center
                    room1_attrs["viz"]["center"] = room_center + self.viz_center_offsets["room"]
                    working_graph.update_node_attrs(room_nodes_ids[0], room1_attrs)

                    combinations = list(itertools.product(room1_ws_nodes_ids, room2_ws_nodes_ids))
                    for combination in combinations:
                        x = segments_distance(working_graph.get_attributes_of_node(combination[0])["limits"],working_graph.get_attributes_of_node(combination[1])["limits"])
                        working_graph.add_edges([(combination[0], combination[1], {"type": "ws_same_room", "x":x, "viz_feat": "b", "linewidth":1.0, "alpha":0.5})])

                    working_graph.remove_nodes(node_ids_to_remove)
        return working_graph


    def reintroduce_predicted_edges(self, unparented_base_graph, predictions, image_name = "name not provided"):
        unparented_base_graph = copy.deepcopy(unparented_base_graph)
        unparented_base_graph.add_edges(predictions)
        # visualize_nxgraph(unparented_base_graph, image_name = image_name)


    def normalize_features_nxdatset(self, nxdatset):
        print(f"SyntheticDatasetGenerator: ", Fore.GREEN + "Normalizing Dataset" + Fore.WHITE)
        generate_x_plots = False
        normalized_nxdatset = {}
        x_history = {"raw": {}, "normalized": {}}
        for tag in nxdatset.keys():
            graphs = []
            for graph in nxdatset[tag]:
                new_graph = copy.deepcopy(graph)
                for node_id in list(new_graph.get_nodes_ids()):
                    node_attrs = new_graph.get_attributes_of_node(node_id)
                    if node_attrs["type"] == "ws":
                        if generate_x_plots:
                            if "nodes" not in x_history["raw"].keys():
                                x_history["raw"]["nodes"] = np.array([node_attrs["x"]])
                            else:
                                x_history["raw"]["nodes"] = np.concatenate([x_history["raw"]["nodes"], np.array([node_attrs["x"]])],axis=0)

                        new_node_attrs = copy.deepcopy(node_attrs)
                        # print(f"dbg node_attrs[x] {node_attrs['x']}")
                        new_node_attrs["x"] = self.normalize_features("node", node_attrs["x"])
                        new_graph.update_node_attrs(node_id, new_node_attrs)
                        if generate_x_plots:
                            if "nodes" not in x_history["normalized"].keys():
                                x_history["normalized"]["nodes"] = np.array([new_node_attrs["x"]])
                            else:
                                x_history["normalized"]["nodes"] = np.concatenate([x_history["normalized"]["nodes"], np.array([new_node_attrs["x"]])],axis=0)

                for edge_id in list(new_graph.get_edges_ids()):
                    edge_attrs = new_graph.get_attributes_of_edge(edge_id)
                    if generate_x_plots:
                        if "edges" not in x_history["raw"].keys():
                            x_history["raw"]["edges"] = np.array([edge_attrs["x"]])
                        else:
                            x_history["raw"]["edges"] = np.concatenate([x_history["raw"]["edges"], np.array([edge_attrs["x"]])],axis=0)

                    edge_attrs["x"] = self.normalize_features("edge", edge_attrs["x"])

                    new_graph.update_edge_attrs(edge_id, edge_attrs)
                    if generate_x_plots:
                        if "edges" not in x_history["normalized"].keys():
                            x_history["normalized"]["edges"] = np.array([edge_attrs["x"]])
                        else:
                            x_history["normalized"]["edges"] = np.concatenate([x_history["normalized"]["edges"], np.array([edge_attrs["x"]])],axis=0)

                graphs.append(new_graph)
            normalized_nxdatset[tag] = graphs
            
        if generate_x_plots:
            self.plot_input_histograms(x_history)
        return normalized_nxdatset


    def plot_input_histograms(self, x_history):
        # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
        sns.set(style="darkgrid")
        # fig, axs = plt.subplots(2, 3, figsize=(14, 14))
        fig = plt.figure(constrained_layout=True)
        fig.suptitle('Nodes histogram')
        subfigs = fig.subfigures(nrows=2, ncols=1)

        subfigs[0].suptitle(f'Raw')
        axs = subfigs[0].subplots(nrows=1, ncols=3)
        sns.histplot(data=x_history["raw"]["nodes"][:,0],  kde=True, color="skyblue", ax=axs[0])
        axs[0].set_title("Length")
        sns.histplot(data=x_history["raw"]["nodes"][:,1], kde=True, color="olive", ax=axs[1])
        axs[1].set_title("Normal X")
        sns.histplot(data=x_history["raw"]["nodes"][:,2],  kde=True, color="gold", ax=axs[2])
        axs[2].set_title("Normal Y")

        subfigs[1].suptitle(f'Normalized')
        axs = subfigs[1].subplots(nrows=1, ncols=3)
        sns.histplot(data=x_history["normalized"]["nodes"][:,0], kde=True, color="skyblue", ax=axs[0])
        axs[0].set_title("Length")
        sns.histplot(data=x_history["normalized"]["nodes"][:,1], kde=True, color="olive", ax=axs[1])
        axs[1].set_title("Normal X")
        sns.histplot(data=x_history["normalized"]["nodes"][:,2], kde=True, color="gold", ax=axs[2])
        axs[2].set_title("Normal Y")

        plt.savefig(os.path.join(self.report_path, "Nodes histogram.png"), bbox_inches='tight')

        fig = plt.figure(constrained_layout=True)
        fig.suptitle('Edges histogram')
        subfigs = fig.subfigures(nrows=2, ncols=1)

        subfigs[0].suptitle(f'Raw')
        axs = subfigs[0].subplots(nrows=1, ncols=3)
        sns.histplot(data=x_history["raw"]["edges"][:,0],  kde=True, color="skyblue", ax=axs[0])
        axs[0].set_title("min(distance)")
        sns.histplot(data=x_history["raw"]["edges"][:,1], kde=True, color="olive", ax=axs[1])
        axs[1].set_title("Relat. position X")
        sns.histplot(data=x_history["raw"]["edges"][:,2],  kde=True, color="gold", ax=axs[2])
        axs[2].set_title("Relat. position Y")

        subfigs[1].suptitle(f'Normalized')
        axs = subfigs[1].subplots(nrows=1, ncols=3)
        sns.histplot(data=x_history["normalized"]["edges"][:,0], kde=True, color="skyblue", ax=axs[0])
        axs[0].set_title("min(distance)")
        sns.histplot(data=x_history["normalized"]["edges"][:,1], kde=True, color="olive", ax=axs[1])
        axs[1].set_title("Relat. position X")
        sns.histplot(data=x_history["normalized"]["edges"][:,2], kde=True, color="gold", ax=axs[2])
        axs[2].set_title("Relat. position Y")

        plt.savefig(os.path.join(self.report_path, "Edges histogram.png"), bbox_inches='tight')


    def dataset_to_hdata(self, nxdataset):
        hdataset = {}
        for key in nxdataset.keys():
            nxdatset_key = nxdataset[key]
            hdataset_key = []
            for nxgraph in nxdatset_key:
                hdataset_key.append(nxgraph.nx_to_hetero())
            hdataset[key] = hdataset_key
        return hdataset
    
    def save_wrappers_to_pickle(self, nxdataset, path):
        import pickle
        print(f"SyntheticDatasetGenerator: ", Fore.GREEN + "Saving Dataset to pickle" + Fore.WHITE)
        with open(path, 'wb') as f:
            pickle.dump(nxdataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"SyntheticDatasetGenerator: ", Fore.GREEN + "Dataset saved to pickle" + Fore.WHITE)

    def save_networkx_graphs_to_pickle(self, nxdataset, path, save_wrapper=False):
        import pickle
        
        # Handle different data structures
        if isinstance(nxdataset, dict):
            # If nxdataset is a dictionary, process each value
            data_to_save = {}
            for key, value in nxdataset.items():
                data_to_save[key] = self._process_dataset_element(value, save_wrapper)
        elif isinstance(nxdataset, list):
            # If nxdataset is a list
            data_to_save = self._process_dataset_element(nxdataset, save_wrapper)
        else:
            # Single element
            data_to_save = self._process_dataset_element(nxdataset, save_wrapper)

        print(f"SyntheticDatasetGenerator: ", Fore.GREEN + "Saving Dataset to pickle" + Fore.WHITE)
        with open(path, 'wb') as f:
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"SyntheticDatasetGenerator: ", Fore.GREEN + "Dataset saved to pickle" + Fore.WHITE)
    
    def _process_dataset_element(self, element, save_wrapper):
        """Helper method to process dataset elements based on their structure"""
        if isinstance(element, list):
            # If element is a list, process each item in the list
            if save_wrapper:
                return element  # Return wrappers as-is
            else:
                return [item.graph if hasattr(item, 'graph') else item for item in element]
        else:
            # Single element
            if save_wrapper:
                return element  # Return wrapper as-is
            else:
                return element.graph if hasattr(element, 'graph') else element

    def serialize_dataset(self, digraphs=False):
        dataset_dir = Path(self.dataset_path)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for dataset_tag, graph_list in self.graphs.items():
            dataset_tag_dir = dataset_dir / dataset_tag
            dataset_tag_dir.mkdir(parents=True, exist_ok=True)

            for i, graph in enumerate(graph_list):
                if digraphs:
                    graph.serialize_diGraph(dataset_tag_dir / f"{i}.pt")
                else:
                    graph.serialize(dataset_tag_dir / f"{i}.pt")

    
    def deserialize_dataset(self, digraphs=False, path = None, number = -1):
        self.graphs["original"].clear()
        self.graphs["noise"].clear()
        self.graphs["extended"].clear()
        
        if path is not None:
            self.dataset_path = path
        dataset_dir = Path(self.dataset_path) 
        
        for dataset_tag, graph_list in self.graphs.items():
            dataset_tag_dir = dataset_dir / dataset_tag 

            counter = 0
            for file in sorted(dataset_tag_dir.glob("*.pt")):
                if counter == number:
                    break
                counter += 1
                
                GraphW = GraphWrapper()  
                if digraphs:
                    graph = nx.DiGraph()
                    graph = GraphW.deserialize_diGraph(str(file))
                    graph_list.append(graph)
                else:
                    GraphW.deserialize(str(file)) 
                    graph_list.append(GraphW)


    def save_pickle(object, filename):
        """Saves a pickled file."""
        with open(filename, 'wb') as f:
            pickle.dump(object, f)
        f.close()

    def load_pickle(filename):
        """
        Loads a pickled file.
        """
        with open(filename, 'rb') as f:
            object = pickle.load(f)
            f.close()
        return object

    def dataset_from_msd(self, path):
        with open(path, 'rb') as f:
            raw_msd_graphs = pickle.load(f)
            print(f"Loaded {len(raw_msd_graphs)} graphs from {path}")
            f.close()

        # get settings limit or default to 100
        msd_limit = self.settings["source"].get("limit", len(raw_msd_graphs))

        graphs = []
        for graph in tqdm.tqdm(raw_msd_graphs[:msd_limit], desc="Processing MSD graphs", colour="red"):
            if graph.number_of_nodes() <= self.settings["source"].get("max_nodes", 9999):
                graphs.append(self.graph_from_msd(GraphWrapper(graph_obj = copy.deepcopy(graph))))

        self.graphs["original"] = graphs

        return graphs

    def graph_from_msd(self, msd_graph):

        graph = copy.deepcopy(msd_graph)

        nodes_attrs = graph.get_attributes_of_all_nodes()
        edges_attrs = graph.get_attributes_of_all_edges()

        nodes_to_remove = []
        edges_to_add = []
        current_node_id = 0
        node_id_mapping = {}
        graph.graph.graph.clear()

        for node_id, node_attrs in nodes_attrs:
            node_id_mapping[node_id] = copy.deepcopy(current_node_id)
            current_node_id += 1
            node_attrs["viz"] = {}
            if node_attrs["type"] in ["room", "wall","floor","building"]:
                node_attrs["center"] = np.array([node_attrs["center"][0], node_attrs["center"][1], 0.])
                node_attrs["viz"]["center"] = copy.deepcopy(node_attrs["center"])
                node_attrs["viz"]["center"][2] = self.viz_center_offsets[node_attrs["type"]][2]
                node_attrs["viz"]["type"] = "Point"
                node_attrs["viz"]["feat"] = self.node_viz_feat_mapping[node_attrs["type"]]
                node_attrs["linewidth"] = 1.0
                node_attrs["alpha"] = 0.5

            elif node_attrs["type"] in ["ws"]:
                node_attrs["center"] = np.array([node_attrs["center"][0], node_attrs["center"][1], 0.])
                node_attrs["normal"] = np.array(node_attrs["normal"])
                node_attrs["length"] = node_attrs["width"]
                rotation = R.from_euler('z', -90, degrees=True)
                ws_direction = rotation.apply(node_attrs["normal"])
                ws_direction /= np.linalg.norm(ws_direction)
                limits = [node_attrs["center"] + ws_direction*node_attrs["length"]/2,
                          node_attrs["center"] - ws_direction*node_attrs["length"]/2]
                node_attrs["limits"] = limits
                limits[0][2] = self.viz_center_offsets[node_attrs["type"]][2]
                limits[1][2] = self.viz_center_offsets[node_attrs["type"]][2]
                node_attrs["viz"]["limits"] = copy.deepcopy(limits)
                node_attrs["viz"]["center"] = copy.deepcopy(node_attrs["center"])
                node_attrs["viz"]["center"][2] = self.viz_center_offsets[node_attrs["type"]][2]
                node_attrs["viz"]["type"] = "Line"
                node_attrs["viz"]["feat"] = self.node_viz_feat_mapping[node_attrs["type"]]
                node_attrs["viz"]["linewidth"] = 2.0
                node_attrs["viz"]["alpha"] = 1.0

            elif node_attrs["type"] in ["wall_ws"]:
                room_id = list(copy.deepcopy(graph).get_neighbourhood_graph(node_id).filter_graph_by_node_types(["room"]).get_nodes_ids())[0]
                WALL_id = list(copy.deepcopy(graph).get_neighbourhood_graph(node_id).filter_graph_by_node_types(["wall"]).get_nodes_ids())[0]
                room_WSs_ids = list(copy.deepcopy(graph).get_neighbourhood_graph(room_id).filter_graph_by_node_types(["ws"]).get_nodes_ids())
                center_WALL_WS = graph.get_attributes_of_node(node_id)["center"]

                minimum_distance = 9999
                closest_WS = None
                for room_WS_id in room_WSs_ids:
                    center_WS = graph.get_attributes_of_node(room_WS_id)["center"]
                    distance = np.linalg.norm(center_WALL_WS - center_WS)
                    if not closest_WS or (minimum_distance > distance):
                        minimum_distance = distance
                        closest_WS = room_WS_id
                
                # print(f"dbg room_id {room_id} WALL_id {WALL_id} closest_WS {closest_WS}")
                nodes_to_remove.append(node_id)
                edges_to_add.append((WALL_id, closest_WS, {"type": "ws_belongs_wall", "linewidth":1.0, "alpha":0.5}))

            else:
                nodes_to_remove.append(node_id)

        for src_id, trg_id, edge_attrs in edges_attrs:
            edge_attrs["linewidth"] = 0.5
            edge_attrs["alpha"] = 0.5

        graph.remove_nodes(nodes_to_remove)
        graph.add_edges(edges_to_add)
        graph.relabel_nodes(node_id_mapping)

        nodes_attrs = graph.get_attributes_of_all_nodes()
        nodes_to_remove = []
        for node_id, node_attrs in nodes_attrs:
            if node_attrs["type"] in ["wall"]:
                ws_count = len(list(copy.deepcopy(graph).get_neighbourhood_graph(node_id).filter_graph_by_node_types(["ws"]).get_nodes_ids()))
                if ws_count <= 1:
                    nodes_to_remove.append(node_id)

        graph.remove_nodes(nodes_to_remove)

        return graph
    

    def dataset_from_disk(self, path):
        """
        Loads graphs from pkl files from folder
        made for "disk" config

        Args:
            - path: path to folder where pkl files are stored
        """

        if path[-4:] == ".pkl" or path[-7:] == ".pickle":
            print("Pickle file path provided, loading single file instead of folder.")
            pkl_files = [Path(path)]

        else:
            folder = Path(path)

            # raise folder errors
            if not folder.exists():
                raise FileNotFoundError(f"Folder not found: {path}")
            if not folder.is_dir():
                raise NotADirectoryError(f"Folder path is not a directory: {path}")
            
            # get both .pkl and .pickle files
            pkl_files = sorted(list(folder.glob("*.pkl")) + list(folder.glob("*.pickle")))

            # raise file error
            if len(pkl_files) == 0:
                raise FileNotFoundError(f"No .pkl/.pickle files found in folder: {path}")
            
        # load limit, same as msd if not set in config
        load_limit = self.settings["source"].get("limit", 100)
        print(f"Loading up to {load_limit} graphs from disk.")
    
        graphs = []
        total_loaded_raw = 0

        for pkl_file in pkl_files:
            with open(pkl_file, "rb") as f:
                obj = pickle.load(f)

            # since each file can contain multiple graphs, handle both
            if isinstance(obj, list):
                raw_graphs = obj
            else:
                raw_graphs = [obj]

            total_loaded_raw += len(raw_graphs)

            for graph in raw_graphs:
                if load_limit is not None and len(graphs) >= load_limit:
                    break

                # wrap as GW 
                if isinstance(graph, GraphWrapper):
                    gw_obj = copy.deepcopy(graph)
                else:
                    gw_obj = GraphWrapper(graph_obj=copy.deepcopy(graph))

                disk_mode = self.settings["source"].get("disk_mode", "passthrough")

                if disk_mode == "passthrough":
                    graphs.append(copy.deepcopy(gw_obj))
                elif disk_mode == "msd":
                    graphs.append(self.graph_from_msd(gw_obj))
                else:
                    raise NotImplementedError("Disk mode not implemented, choose 'passthrough' or 'msd'")

            if load_limit is not None and len(graphs) >= load_limit:
                break

        print(f"Loaded {len(graphs)} graphs from folder {path} (Total discovered files: {len(pkl_files)}, with {total_loaded_raw} raw graphs).")

        self.graphs["original"] = graphs
        return graphs

    
    def add_complete_viz_attributes(self):
                
        for key in self.graphs.keys():
            for i in range(len(self.graphs[key])):
                self.graphs[key][i].add_complete_viz_attributes_to_graph(self.graphs[key][i], self.viz_center_offsets, self.node_viz_feat_mapping)



    def deserialize_and_transform_to_GWraph(self):
        msd_dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),"msd")
        msd_dataset_dir = "/home/adminpc/workspaces/reasoning_ws/src/msd/"

        sys.path.append(msd_dataset_dir)

        self.graphs["original"].clear()
        self.graphs["noise"].clear()
        self.graphs["extended"].clear()

        dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),"msd/data") 

        dim_name = os.path.join(dataset_dir, "MSD - Apartment-Level partials graphs counter cleaned.pickle")
        graph_name = os.path.join(dataset_dir, "MSD - Apartment-Level partials graphs cleaned 5.0K.pickle")

        graphs = []
        dimensions = []
        # load the dataset
        with open(dim_name, 'rb') as f:
            dimensions = pickle.load(f)
            f.close()
        
        with open(graph_name, 'rb') as f:
            graphs = pickle.load(f)
            f.close()

        #convert all objects to GraphWrapper and insert them insiede original
        for i in range(len(dimensions)):
            for j in range(dimensions[i]):
                graph = GraphWrapper()
                graph.graph.add_nodes_from(graphs[i][j].nodes(data=True))
                graph.graph.add_edges_from(graphs[i][j].edges(data=True))
                self.graphs["noise"].append(graph)
                if j == dimensions[i] - 1:
                    self.graphs["original"].append(graph)
        
        return dimensions

    #serialize graphs back in original an noise folder
    def serialize_MSD_dataset(self, dimensions):
        dataset_dir = Path(self.dataset_path)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for dataset_tag, graph_list in self.graphs.items():
            dataset_tag_dir = dataset_dir / dataset_tag
            dataset_tag_dir.mkdir(parents=True, exist_ok=True)

            if dataset_tag == "original" or dataset_tag == "extended":
                for i, graph in enumerate(graph_list):
                    graph.serialize_diGraph(dataset_tag_dir / f"{i}.pt")
                    
            # uncomment for partial graph matching        
            # if dataset_tag == "noise" or dataset_tag == "extended":
            #     idx = 0  # starting position
            #     for group_index, size in enumerate(dimensions):
            #         for i in range(1, size + 1):
            #             graph = graph_list[idx]
            #             graph.serialize_diGraph(dataset_tag_dir / f"{group_index}_{i}.pt")
            #             idx += 1
            
        # serilize in the dataset_dir the dimensions
        with open(os.path.join(dataset_dir, "dimensions.pickle"), 'wb') as f:
            pickle.dump(dimensions, f)
            f.close()


    def deserialize_MSD_dataset(self):
        dataset_dir = Path(self.dataset_path)

        # Load dimensions
        dimensions_file = dataset_dir / "dimensions.pickle"
        if not dimensions_file.exists():
            raise FileNotFoundError(f"Dimensions file not found at {dimensions_file}")
        with open(dimensions_file, 'rb') as f:
            dimensions = pickle.load(f)

        # Clear existing graphs
        self.graphs["original"].clear()
        self.graphs["noise"].clear()
        self.graphs["extended"].clear()

        def extract_numeric_key(file):
            """Extracts (X, Y) from filenames like 'X_Y.pt' for proper numeric sorting."""
            name_parts = file.stem.split("_")
            return int(name_parts[0]), int(name_parts[1])

        original_dir = dataset_dir / "original"
        original_files = sorted(original_dir.glob("*.pt"), key=lambda f: int(f.stem))

        for file in original_files:
            graph = GraphWrapper()
            graph.deserialize_diGraph(str(file))
            self.graphs["original"].append(graph)

        extended_dir = dataset_dir / "extended"
        extended_files = sorted(extended_dir.glob("*.pt"), key=extract_numeric_key)

        for file in extended_files:
            graph = GraphWrapper()
            graph.deserialize_diGraph(str(file))
            self.graphs["extended"].append(graph)

        return dimensions

    # print all the graphs inside the dataset
    def print_dataset(self):
        for dataset_tag in self.graphs.keys():
            print(f"Dataset tag: {dataset_tag}")
            for i, data in enumerate(self.graphs[dataset_tag]):
                print(f"Graph {i}")
                data.print_attributes()
                print("\n")

    def compute_stats(self, graphs):
        node_counts = []
        edge_counts = []
        node_type_totals = defaultdict(list)
        degree_counts = []

        for G in graphs:
            node_counts.append(len(G.nodes))
            edge_counts.append(len(G.edges))

            # Count node types
            type_counter = Counter()
            for _, data in G.nodes(data=True):
                node_type = data.get("type", "unknown")
                type_counter[node_type] += 1
            for t, count in type_counter.items():
                node_type_totals[t].append(count)

            # Degree distribution
            degrees = [d for _, d in G.degree()]
            degree_counts.extend(degrees)

        avg_nodes = np.mean(node_counts)
        max_nodes = np.max(node_counts)

        avg_edges = np.mean(edge_counts)
        max_edges = np.max(edge_counts)

        avg_node_types = {k: np.mean(v) for k, v in node_type_totals.items()}
        max_node_types = {k: np.max(v) for k, v in node_type_totals.items()}

        return {
            "avg_nodes": avg_nodes,
            "max_nodes": max_nodes,
            "avg_edges": avg_edges,
            "max_edges": max_edges,
            "avg_node_types": avg_node_types,
            "max_node_types": max_node_types,
            "degree_distribution": degree_counts,
        }


    # def merge_graphs_type_as_x(self, nxdatset):
    #     # Initialize lists for concatenated features
    #     all_x = []
    #     all_edge_index = []
    #     all_edge_attrs = []
    #     all_y = []
    #     all_idx = []

    #     # Initialize the slices dictionary
    #     slices = {'x': [0], 'edge_index': [0], 'edge_attr': [0], "y": [], "idx": []}
        
    #     node_offset = 0
    #     edge_offset = 0

    #     for nxgraph in nxdatset:
    #         # nxgraph.to_undirected()
    #         graph = nxgraph.nx_to_homo()

    #         # Append node features and update slices for x
    #         all_x.append(graph.node_type)
    #         slices['x'].append(slices['x'][-1] + graph.x.size(0))

    #         # Append edge indices (shifted by current node offset) and update slices for edge_index
    #         # all_edge_index.append(graph.edge_index + node_offset)
    #         inverse_edge_index = torch.Tensor(np.array([graph.edge_index[1],graph.edge_index[0]])).int()
    #         all_edge_index.append(graph.edge_index)
    #         all_edge_index.append(inverse_edge_index)
    #         slices['edge_index'].append(slices['edge_index'][-1] + graph.edge_index.size(1)*2)

    #         # Append edge attributes and update slices for edge_attrs
    #         all_edge_attrs.append(graph.edge_type)
    #         all_edge_attrs.append(graph.edge_type)
    #         slices['edge_attr'].append(slices['edge_attr'][-1] + graph.edge_attr.size(0)*2)

    #         # Update node and edge offsets
    #         node_offset += graph.x.size(0)
    #         edge_offset += graph.edge_index.size(1)

    #         # Update y and edx
    #         all_y.append(0.)
    #         all_idx.append(len(all_idx))
    #         slices['y'].append(len(slices['y']))
    #         slices['idx'].append(len(slices['idx']))

        

    #     slices['x'], slices['edge_index'], slices['edge_attr'] = torch.Tensor(slices['x']).int(), torch.Tensor(slices['edge_index']).int(), torch.Tensor(slices['edge_attr']).int()
    #     slices['y'].append(len(slices['y']))
    #     slices['y'] = torch.Tensor(slices['y']).int() 
    #     slices['idx'].append(len(slices['idx']))
    #     slices['idx'] = torch.Tensor(slices['idx']).int()
        
    #     # Concatenate all the individual parts
    #     x = torch.cat(all_x, dim=0)
    #     edge_index = torch.cat(all_edge_index, dim=1)
    #     edge_attrs = torch.cat(all_edge_attrs, dim=0)
    #     y = torch.Tensor(all_y)
    #     idx = torch.Tensor(all_idx)

    #     # Create a new Data object with the concatenated features
    #     merged_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attrs, y = y, idx = idx)
    #     # print(f"dbg len(merged_graph.edge_index) {merged_graph.edge_index[:, -10:]}")
    #     # merged_graph.edge_index = to_undirected(merged_graph.edge_index)
    #     # print(f"dbg len(merged_graph.edge_index) {merged_graph.edge_index[:, -10:]}")
    #     # asdf

    #     return merged_graph, slices