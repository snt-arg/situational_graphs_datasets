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
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
import os
import ast

# graph_wrapper_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_wrapper")
# sys.path.append(graph_wrapper_dir)
from graph_wrapper.GraphWrapper import GraphWrapper
# graph_datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_datasets")
# sys.path.append(graph_datasets_dir)
from graph_datasets.graph_visualizer import visualize_nxgraph
from graph_datasets.NodeEdgeFeatureEmbeddingBuildier import NodeEdgeFeatureEmbeddingBuildier
# graph_matching_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_matching")
# sys.path.append(graph_matching_dir)
from graph_matching.utils import relative_positions, segments_distance, closest_point_on_segment, distance_between_points, are_segments_collinear, relative_geometry
# graph_reasoning_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_reasoning")
# sys.path.append(graph_reasoning_dir)


class SyntheticDatasetGenerator():

    def __init__(self, settings, logger = None, report_path = "", dataset_name = ""):
        print(f"SyntheticDatasetGenerator:", Fore.GREEN + "Initializing" + Fore.WHITE)
        self.settings = self.correct_json_initfeat_keys(settings)
        self.logger = logger
        self.report_path = report_path
        self.dataset_name = dataset_name
        self.define_norm_limits()

    def correct_json_initfeat_keys(self, settings):
        new_settings = copy.deepcopy(settings)
        for key, value in settings["initial_features"]["edges"].items():
            new_key = tuple([item.strip() for item in key.strip("[]").split(",")])
            new_settings["initial_features"]["edges"][new_key] = value
            new_settings["initial_features"]["edges"].pop(key)

        return new_settings

    def define_norm_limits(self):
        grid_dims = self.settings["base_graphs"]["grid_dims"]
        max_grid_dims = max(grid_dims[0][1],grid_dims[1][1])
        max_room_center_distances = self.settings["base_graphs"]["room_center_distances"][-1]
        max_room_entry_size = self.settings["base_graphs"]["max_room_entry_size"][-1]
        init_feat_keys = self.settings["initial_features"]
        max_building_size = max_grid_dims*max_room_center_distances
        max_room_size = max_room_entry_size*max_room_center_distances
        max_wall_thickness = grid_dims = self.settings["base_graphs"]["wall_thickness"][1]

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

    def normalize_features(self, type, feats):
        if len(feats) != 0:
            feats_norm = (feats-self.norm_limits[type]["min"])/(self.norm_limits[type]["max"]-self.norm_limits[type]["min"])
        else:
            feats_norm = []
        return feats_norm

    def create_dataset(self):
        print(f"SyntheticDatasetGenerator: ", Fore.GREEN + "Generating Syntetic Dataset" + Fore.WHITE)
        n_buildings = self.settings["base_graphs"]["n_buildings"]

        self.graphs = {"original":[],"noise":[],"views":[]}
        self.max_n_rooms = 0

        def process_building(_):
            base_matrix = self.generate_base_matrix()
            # original_graph = self.generate_graph_from_base_matrix(base_matrix, add_noise=False)
            original_graph = None
            noisy_graph = self.generate_graph_from_base_matrix(base_matrix, add_noise=True)
            return original_graph, noisy_graph

        # Using ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=1) as executor:
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
        grid_dims = [np.random.randint(self.settings["base_graphs"]["grid_dims"][0][0], self.settings["base_graphs"]["grid_dims"][0][1] + 1),
                     np.random.randint(self.settings["base_graphs"]["grid_dims"][1][0], self.settings["base_graphs"]["grid_dims"][1][1] + 1)]
        max_room_entry_size = np.random.randint(self.settings["base_graphs"]["max_room_entry_size"][0], self.settings["base_graphs"]["max_room_entry_size"][1] + 1)
        min_room_entry_size = np.random.randint(self.settings["base_graphs"]["min_room_entry_size"][0], self.settings["base_graphs"]["min_room_entry_size"][1] + 1)

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
                    room_entry_size = [min(remaining[0], np.random.randint(low=min_room_entry_size+1, high=max_room_entry_size+1, size=(1))[0]),\
                                       min(remaining[1], np.random.randint(low=min_room_entry_size+1, high=max_room_entry_size+1, size=(1))[0])]

                    if (room_entry_size[0] >= min_room_entry_size) & (room_entry_size[1] >= min_room_entry_size):
                        room_id = room_n
                        room_n += 1
                    else:
                        room_id = -1
                    for ii in range(room_entry_size[0]):
                        for jj in range(room_entry_size[1]):
                            base_matrix[i+ii, j+jj] = room_id
        self.max_n_rooms = max(self.max_n_rooms, room_n)
        return base_matrix


    def generate_graph_from_base_matrix(self, base_matrix, add_noise = False, add_multiview = False):
        graph = GraphWrapper()
        room_center_distances = self.settings["base_graphs"]["room_center_distances"]
        wall_thickness = np.random.uniform(self.settings["base_graphs"]["wall_thickness"][0], self.settings["base_graphs"]["wall_thickness"][1])

        if add_noise:
            if self.settings["noise"]["global"]["active"]:
                noise_global_center = np.concatenate([np.array(self.settings["base_graphs"]["playground_size"]) * self.settings["noise"]["global"]["translation"] * (np.random.rand(2)- 0.5), [0]])
                noise_global_rotation_angle = (np.random.rand(1)*360*self.settings["noise"]["global"]["rotation"])[0]
            else:
                noise_global_center = [0,0,0]
                noise_global_rotation_angle = 0

        ### Rooms
        room_ids = np.unique(base_matrix)
        room_ids = np.delete(room_ids, np.where(room_ids == -1))
        for base_matrix_room_id in room_ids:
            occurrencies = np.argwhere(np.where(base_matrix == base_matrix_room_id, True, False))
            limits = [occurrencies[0],occurrencies[-1]]
            room_entry_size = [limits[1][0] - limits[0][0] + 1, limits[1][1] - limits[0][1] + 1]
            node_ID = max(graph.get_nodes_ids(), default=-1) + 1
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
            
            graph.add_nodes([(node_ID,{"type" : "room","center" : room_center, "x": room_center, "orientation_angle": room_orientation_angle, "area" : room_area, "Geometric_info" : geometric_info,\
                                            "viz_type" : "Point", "viz_data" : room_center[:2], "viz_feat" : 'ro'})])
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
                ws_normal = np.array([-1,-1, 0])*normals[i]
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

                graph.add_nodes([(node_ID,{"type" : "ws","center" : ws_center, "y" : y, "normal" : ws_normal, "Geometric_info" : geometric_info,\
                                           "viz_type" : "Line", "viz_data" : [ws_limit_1[:2],ws_limit_2[:2]], "viz_feat" : color_map[i],\
                                           "canonic_normal_index" : canonic_normals[i], "linewidth": 2.0, "limits": [ws_limit_1,ws_limit_2],
                                           "length": ws_length})])
                graph.add_edges([(node_ID, node_data[0], {"type": "ws_belongs_room", "x": [], "viz_feat" : 'red', "linewidth":1.0, "alpha":0.5})])
                
                ### Fully connected version
                for prior_ws_i in range(i):
                    x = segments_distance(graph.get_attributes_of_node(node_ID)["limits"],graph.get_attributes_of_node(node_ID-(prior_ws_i+1))["limits"])
                    graph.add_edges([(node_ID, node_ID-(prior_ws_i+1), {"type": "ws_same_room", "x":x, "viz_feat": "b", "linewidth":1.0, "alpha":0.5})])
                # ### Only consecutive wall surfaces
                # if i > 0:
                #     graph.add_edges([(node_ID, node_ID - 1, {"type": "ws_same_room", "viz_feat": "b", "linewidth":1.0, "alpha":0.5})])
                # if i == 3:
                #     graph.add_edges([(node_ID, node_ID - 3, {"type": "ws_same_room", "viz_feat": "b", "linewidth":1.0, "alpha":0.5})])
                # ### Only opposite wall surfaces
                # if i > 1:
                #     graph.add_edges([(node_ID, node_ID - 2, {"type": "ws_same_room", "viz_feat": "b", "linewidth":1.0, "alpha":0.5})])
                ###

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
                    if current_room_id != -1.0 and comparison.all() and current_room_id != base_matrix[compared_ij[0],compared_ij[1]]:
                        compared_room_id = base_matrix[compared_ij[0],compared_ij[1]]
                        if compared_room_id != -1.0 and (current_room_id, compared_room_id) not in explored_walls:
                            explored_walls.append((current_room_id, compared_room_id))
                            graph.to_directed()
                            current_room_neigh = graph.get_neighbourhood_graph(current_room_id-1).filter_graph_by_node_types(["ws"])
                            current_room_neigh_ws_id = list(current_room_neigh.filter_graph_by_node_attributes({"canonic_normal_index" : ij_difference_3D}).get_nodes_ids())[0]
                            current_room_neigh_ws_center = current_room_neigh.get_attributes_of_node(current_room_neigh_ws_id)["center"]

                            compared_room_neigh = graph.get_neighbourhood_graph(compared_room_id-1).filter_graph_by_node_types(["ws"])
                            compared_room_neigh = graph.get_neighbourhood_graph(compared_room_id-1).filter_graph_by_node_types(["ws"])
                            ij_difference_3D_oppposite = list(-1*np.array(ij_difference_3D))
                            compared_room_neigh_ws_id = list(compared_room_neigh.filter_graph_by_node_attributes({"canonic_normal_index" : ij_difference_3D_oppposite}).get_nodes_ids())[0]
                            compared_room_neigh_ws_center = compared_room_neigh.get_attributes_of_node(compared_room_neigh_ws_id)["center"]

                            wall_center = np.array(np.array(current_room_neigh_ws_center) + (np.array(compared_room_neigh_ws_center) - np.array(current_room_neigh_ws_center))/2)
                            node_ID = max(graph.get_nodes_ids(), default=-1) + 1
                            graph.add_nodes([(node_ID,{"type" : "wall", "x" : wall_center, "center" : wall_center,"viz_type" : "Point", "viz_data" : wall_center, "viz_feat" : 'mo'})])
                            graph.add_edges([(current_room_neigh_ws_id, node_ID, {"type": "ws_belongs_wall", "x": [], "viz_feat": "m", "linewidth":1.0, "alpha":0.5}),\
                                             (compared_room_neigh_ws_id, node_ID, {"type": "ws_belongs_wall","viz_feat": "m", "x": [], "linewidth":1.0, "alpha":0.5})])
                            graph.add_edges([(current_room_neigh_ws_id, compared_room_neigh_ws_id, {"type": "ws_same_wall", "x": [], "viz_feat": "orange", "linewidth":1.0, "alpha":0.5})])
                            if add_multiview:
                                graph.update_node_attrs(node_ID, {"view" : graph.get_attributes_of_node(current_room_neigh_ws_id)["view"]})

        return graph

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
                specific_edge_types = [e[1] for e in full_edge_types]
                filtered_graph = filtered_graph.filter_graph_by_edge_types(specific_edge_types)
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
                    node_ids_selected = []
                    for room_node_id in room_node_ids:
                        left_rooms = list(working_graph.filter_graph_by_node_types(["room"]).get_nodes_ids())
                        if len(left_rooms) > 1 and np.random.random_sample() < pp_settings["room"]:
                            node_ids_selected.append(room_node_id)
                            # asdf
                            ws_node_ids = list(working_graph.get_neighbourhood_graph(room_node_id).filter_graph_by_node_types(["ws"]).get_nodes_ids())
                            node_ids_selected = node_ids_selected + ws_node_ids
                            for ws_node_id in ws_node_ids:
                                wall_ws_node_ids = list(working_graph.get_neighbourhood_graph(ws_node_id).filter_graph_by_node_types(["wall"]).get_nodes_ids())
                                for wall_ws_node_id in wall_ws_node_ids:
                                    if len(list(working_graph.get_neighbourhood_graph(wall_ws_node_id).filter_graph_by_node_types(["ws"]).get_nodes_ids())) < 3:
                                        node_ids_selected.append(wall_ws_node_id)
                        working_graph.remove_nodes(node_ids_selected)

                ### ws dropout
                if pp_settings["ws"] > 0.:
                    ws_node_ids = list(working_graph.filter_graph_by_node_types(["ws"]).get_nodes_ids())
                    node_ids_selected = []
                    for ws_node_id in ws_node_ids:
                        # visualize_nxgraph(working_graph.get_neighbourhood_graph(ws_node_id), "test", visualize_alone=True)
                        # for e in working_graph.get_neighbourhood_graph(ws_node_id).get_attributes_of_all_edges():
                        #     print(f"dbg e[2][type] {e[2]['type']}")
                        # visualize_nxgraph(working_graph.get_neighbourhood_graph(ws_node_id).filter_graph_by_edge_types(["ws_same_room"]).filterout_unparented_nodes(), "test 2", visualize_alone=True)
                        same_room_ws_node_ids = list(working_graph.get_neighbourhood_graph(ws_node_id).filter_graph_by_edge_types(["ws_same_room"]).filterout_unparented_nodes().get_nodes_ids())
                        if len(same_room_ws_node_ids) > 1 and np.random.random_sample() < pp_settings["ws"]:
                            node_ids_selected.append(ws_node_id)
                    working_graph.remove_nodes(node_ids_selected)

            ### Include K nearest neighbouors edges
            elif pp_settings["pp_name"] == "K_near_neigh":
                node_ids = list(working_graph.filter_graph_by_node_types(pp_settings["types"]).get_nodes_ids())
                centers = np.array([working_graph.get_attributes_of_node(node_id)["center"] for node_id in node_ids])
                kdt = KDTree(centers, leaf_size=30, metric='euclidean')
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
                                ws_attrs["viz_data"] = new_limits
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
                                new_ws_attrs["viz_data"] = new_limits
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
                fig = visualize_nxgraph(working_graph, pp_settings["fig_name"], visualize_alone=pp_settings["visualize_alone"])
                if pp_settings["save_path"]:
                    fig.savefig(pp_settings["save_path"], bbox_inches='tight')
            elif pp_settings["pp_name"] == "remove_self_loops":
                working_graph.remove_self_loops()
            elif pp_settings["pp_name"] == "relabel_nodes":
                working_graph.relabel_nodes(mapping = pp_settings["mapping"], copy=pp_settings["copy"])
            elif pp_settings["pp_name"] == "unfreeze":
                working_graph.unfreeze()

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

            if len(base_graph.get_nodes_ids()) > 0 and len(base_graph.get_edges_ids()) > 0:
                new_nxdataset.append(base_graph)

        val_start_index = int(len(nxdataset)*(1-self.settings["training_split"]["val"]-self.settings["training_split"]["test"]))
        test_start_index = int(len(nxdataset)*(1-self.settings["training_split"]["test"]))
        extended_nxdatset = {"train" : new_nxdataset[:val_start_index], "val" : new_nxdataset[val_start_index:test_start_index],"test" : new_nxdataset[test_start_index:-1]}

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
                            ws_node_attrs["viz_data"][min_col] = copy.deepcopy(np.array(other_room_ws_closest_points[min_dist_idx][:2]))
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
                                wall_attrs["viz_data"] = new_wall_center
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
                                ws0_attrs["viz_data"] = list(new_limits)
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
                                        wall_attrs["viz_data"] = new_wall_center
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
                    room1_attrs["viz_data"] = room_center[:2]
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

    # def save_to_files(self):
    #     dataset_dir = graph_datasets_dir + f"/{self.dataset_name}"
    #     if not os.path.exists(dataset_dir):
    #         os.makedirs(dataset_dir)
    #     for dataset_tag in self.graphs.keys():
    #         dataset_tag_dir = dataset_dir + f"/{dataset_tag}"
    #         if not os.path.exists(dataset_tag_dir):
    #             os.makedirs(dataset_tag_dir)
    #         for i, data in enumerate(self.graphs[dataset_tag]):
    #             data.to_file(dataset_tag_dir + f"/{i}.pt")    

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