import numpy as np
import copy
import itertools
import random, math, time
import tqdm
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import KDTree
from colorama import Fore, Back, Style
import seaborn as sns
import matplotlib.pyplot as plt

import sys
import os
from situational_graphs_wrapper.GraphWrapper import GraphWrapper
from situational_graphs_datasets.graph_visualizer import visualize_nxgraph
from situational_graphs_reasoning.utils import relative_positions, segments_distance
from situational_graphs_reasoning.from_networkxwrapper_2_heterodata import from_networkxwrapper_2_heterodata


class SyntheticDatasetGenerator():

    def __init__(self, settings, logger = None, report_path = ""):
        print(f"SyntheticDatasetGenerator:", Fore.GREEN + "Initializing" + Fore.WHITE)
        self.settings = settings
        self.logger = logger
        self.report_path = report_path
        self.define_norm_limits()

    def define_norm_limits(self):
        playground_size = self.settings["base_graphs"]["playground_size"]
        max_room_entry_size = self.settings["base_graphs"]["max_room_entry_size"][-1]
        min_room_entry_size = self.settings["base_graphs"]["min_room_entry_size"][0]
        max_room_center_distances = self.settings["base_graphs"]["room_center_distances"][-1]
        min_room_center_distances = self.settings["base_graphs"]["room_center_distances"][0]
        init_feat_keys = self.settings["initial_features"]
        max_building_size = max_room_entry_size*max_room_center_distances
        min_building_size = min_room_entry_size*min_room_center_distances

        def add_features(type, feature_keys, working_dict):
            if type == "ws_node":
                if feature_keys[0] == "centroid":
                    working_dict["min"] = np.concatenate([working_dict["min"], -np.array(playground_size)/2 - max_building_size])
                    working_dict["max"] = np.concatenate([working_dict["max"], np.array(playground_size)/2 + max_building_size])
                elif feature_keys[0] == "length":
                    working_dict["min"] = np.concatenate([working_dict["min"], [min_building_size]])
                    working_dict["max"] = np.concatenate([working_dict["max"], [max_building_size]]) #, [np.log(max_room_entry_size*max_room_center_distances)]])
                elif feature_keys[0] == "normals":
                    working_dict["min"] = np.concatenate([working_dict["min"],[-1,-1]])
                    working_dict["max"] = np.concatenate([working_dict["max"],[1,1]])

            elif type == "edge":
                if feature_keys[0] == "relative_pos":
                    working_dict["min"] = np.concatenate([working_dict["min"],-np.array([max_building_size,max_building_size])])
                    working_dict["max"] = np.concatenate([working_dict["max"],np.array([max_building_size,max_building_size])])
                elif feature_keys[0] == "min_dist":
                    working_dict["min"] = np.concatenate([working_dict["min"],[0]])
                    working_dict["max"] = np.concatenate([working_dict["max"],[max_building_size]])  #,[np.log(max(playground_size)+1)]])
            
            if len(feature_keys) > 1:
                working_dict = add_features(type, feature_keys[1:], working_dict)
            return working_dict

        self.norm_limits = {"ws_node" : add_features("ws_node", init_feat_keys["ws_node"], {"min": [], "max":[]}), \
                            "edge" : add_features("edge", init_feat_keys["edge"], {"min": [], "max":[]})}

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
        for n_building in tqdm.tqdm(range(n_buildings), colour="green"):
            base_matrix = self.generate_base_matrix()

            self.graphs["original"].append(self.generate_graph_from_base_matrix(base_matrix, add_noise= False))
            self.graphs["noise"].append(self.generate_graph_from_base_matrix(base_matrix, add_noise= True))
            self.graphs["views"].append(self.generate_graph_from_base_matrix(base_matrix, add_noise= False, add_multiview=True))
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
        graph.to_undirected()
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
            node_ID = len(graph.get_nodes_ids())
            room_center = np.array([room_center_distances[0]*(limits[0][0] + (room_entry_size[0]-1)/2), room_center_distances[1]*(limits[0][1]+(room_entry_size[1]-1)/2), 0])
            room_orientation_angle = 0.0
            room_area = [room_center_distances[0]*room_entry_size[0] - wall_thickness, room_center_distances[1]*room_entry_size[1] - wall_thickness, 0]
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
            
            graph.add_nodes([(node_ID,{"type" : "room","center" : room_center, "x": [], "orientation_angle": room_orientation_angle, "area" : room_area, "Geometric_info" : geometric_info,\
                                            "viz_type" : "Point", "viz_data" : room_center[:2], "viz_feat" : 'bo'})])
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
                node_ID = len(graph.get_nodes_ids())
                orthogonal_normal = R.from_euler("Z", 90, degrees= True).apply(copy.deepcopy(normals[i]))
                orthogonal_canonic_normal = R.from_euler("Z", 90, degrees= True).apply(copy.deepcopy(canonic_normals[i]))
                ws_normal = np.array([-1,-1, 0])*normals[i]
                ws_center = node_data[1]["center"] + abs(np.dot(np.array(node_data[1]['area'])/2,canonic_normals[i]))*np.array(normals[i])

                ws_length = abs(np.dot(np.array(node_data[1]['area']),canonic_normals[i]))
                ws_limit_1 = ws_center + abs(np.dot(np.array(node_data[1]['area'])/2,np.array(orthogonal_canonic_normal)))*np.array(orthogonal_normal)
                ws_limit_2 = ws_center + abs(np.dot(np.array(node_data[1]['area'])/2,-np.array(orthogonal_canonic_normal)))*(-np.array(orthogonal_normal))
                
                def add_ws_node_features(feature_keys, feats):
                    if feature_keys[0] == "centroid":
                        feats = np.concatenate([feats, ws_center[:2]]).astype(np.float32)
                    elif feature_keys[0] == "length":
                        feats = np.concatenate([feats, [ws_length]]).astype(np.float32)   #, [np.log(ws_length)]]).astype(np.float32)
                    elif feature_keys[0] == "normals":
                        feats = np.concatenate([feats, ws_normal[:2]]).astype(np.float32)
                    if len(feature_keys) > 1:
                        feats = add_ws_node_features(feature_keys[1:], feats)
                    return feats

                x = add_ws_node_features(self.settings["initial_features"]["ws_node"], [])
                y = int(node_data[0])
                geometric_info = np.concatenate([ws_center, ws_normal])
                color_map = ["green", "orange", "red", "pink"]
                color_map = ["black", "black", "black", "black"]

                graph.add_nodes([(node_ID,{"type" : "ws","center" : ws_center, "x" : x, "y" : y, "normal" : ws_normal, "Geometric_info" : geometric_info,\
                                           "viz_type" : "Line", "viz_data" : [ws_limit_1[:2],ws_limit_2[:2]], "viz_feat" : color_map[i],\
                                           "canonic_normal_index" : canonic_normals[i], "linewidth": 2.0, "limits": [ws_limit_1,ws_limit_2]})])
                graph.add_edges([(node_ID, node_data[0], {"type": "ws_belongs_room", "x": [], "viz_feat" : 'b', "linewidth":1.0, "alpha":0.5})])

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
                            current_room_neigh = graph.get_neighbourhood_graph(current_room_id-1).filter_graph_by_node_types(["ws"])
                            current_room_neigh_ws_id = list(current_room_neigh.filter_graph_by_node_attributes({"canonic_normal_index" : ij_difference_3D}).get_nodes_ids())[0]
                            current_room_neigh_ws_center = current_room_neigh.get_attributes_of_node(current_room_neigh_ws_id)["center"]

                            compared_room_neigh = graph.get_neighbourhood_graph(compared_room_id-1).filter_graph_by_node_types(["ws"])
                            compared_room_neigh = graph.get_neighbourhood_graph(compared_room_id-1).filter_graph_by_node_types(["ws"])
                            ij_difference_3D_oppposite = list(-1*np.array(ij_difference_3D))
                            compared_room_neigh_ws_id = list(compared_room_neigh.filter_graph_by_node_attributes({"canonic_normal_index" : ij_difference_3D_oppposite}).get_nodes_ids())[0]
                            compared_room_neigh_ws_center = compared_room_neigh.get_attributes_of_node(compared_room_neigh_ws_id)["center"]

                            wall_center = np.array(np.array(current_room_neigh_ws_center) + (np.array(compared_room_neigh_ws_center) - np.array(current_room_neigh_ws_center))/2)
                            node_ID = len(graph.get_nodes_ids())
                            graph.add_nodes([(node_ID,{"type" : "wall", "x" : wall_center, "center" : wall_center,"viz_type" : "Point", "viz_data" : wall_center, "viz_feat" : 'co'})])
                            graph.add_edges([(current_room_neigh_ws_id, node_ID, {"type": "ws_belongs_wall", "x": [], "viz_feat": "c", "linewidth":1.0, "alpha":0.5}),\
                                             (compared_room_neigh_ws_id, node_ID, {"type": "ws_belongs_wall","viz_feat": "c", "x": [], "linewidth":1.0, "alpha":0.5})])
                            graph.add_edges([(current_room_neigh_ws_id, compared_room_neigh_ws_id, {"type": "ws_same_wall", "x": [], "viz_feat": "c", "linewidth":1.0, "alpha":0.5})])
                            if add_multiview:
                                graph.update_node_attrs(node_ID, {"view" : graph.get_attributes_of_node(current_room_neigh_ws_id)["view"]})

        ### Floors
        rooms_attrs = graph.filter_graph_by_node_attributes({"type" : "room"}).get_attributes_of_all_nodes()
        room_ids = [attr[0] for attr in rooms_attrs]
        room_centers = [attr[1]["center"] for attr in rooms_attrs]
        floor_center = np.array(room_centers).sum(axis=0) / len(room_centers)
        floor_node_id = len(graph.get_nodes_ids())
        graph.add_nodes([(floor_node_id,{"type" : "floor", "x" : floor_center, "center" : floor_center,\
                            "viz_type" : "Point", "viz_data" : floor_center[:2], "viz_feat" : 'oC1'})])
        for room_id in room_ids:
            graph.add_edges([(room_id, floor_node_id, {"type": "room_belongs_floor", "x": [],"viz_feat": "orange",\
                                                        "linewidth":1.0, "alpha":0.5})])

        return graph
    
    def set_dataset(self, tag, nxdata):
        self.graphs[tag] = nxdata
    
    def get_filtered_datset(self, node_types, edge_types):
        print(f"SyntheticDatasetGenerator: ", Fore.GREEN + "Filtering Dataset" + Fore.WHITE)
        nx_graphs = {}
        for key in self.graphs.keys():
            nx_graphs_key = []
            for base_graph in self.graphs[key]:
                filtered_graph = base_graph.filter_graph_by_node_types(node_types)
                filtered_graph.relabel_nodes() ### TODO What to do when Im dealing with different node types? Check tutorial
                filtered_graph = filtered_graph.filter_graph_by_edge_types([edge_type for edge_type in edge_types])
                nx_graphs_key.append(filtered_graph)
            nx_graphs[key] = nx_graphs_key

        return nx_graphs
    

    def extend_nxdataset(self, nxdataset, new_edge_types, stage):
        print(f"SyntheticDatasetGenerator: ", Fore.GREEN + "Extending Dataset" + Fore.WHITE)
        new_nxdataset = []

        # for i in tqdm.tqdm(range(len(nxdataset)), colour="green"):
        for i in range(len(nxdataset)):
            nxdata = nxdataset[i]
            base_graph = copy.deepcopy(nxdata)
            positive_gt_edge_ids = list(base_graph.get_edges_ids())
            settings = self.settings["postprocess"][stage]
            base_graph.unfreeze()
            
            ### Set positive label
            if settings["use_gt"]:
                for edge_id in list(base_graph.get_edges_ids()):
                    base_graph.update_edge_attrs(edge_id, {"label":1, "viz_feat" : 'green'})
            else:
                base_graph.remove_all_edges()
            base_graph.to_directed()
                
            ### NODE DROPOUT
            if settings["node_dropout"] > 0.:
                node_ids = list(base_graph.get_nodes_ids())
                node_ids_selected = []
                for node_id in node_ids:
                    if np.random.random_sample() < settings["node_dropout"]:
                        node_ids_selected.append(node_id)
                base_graph.remove_nodes(node_ids_selected)

            ### Include K nearest neighbouors edges
            if settings["K_nearest_max"] > 0:
                node_ids = list(base_graph.filter_graph_by_node_types(settings["K_nearest_types"]).get_nodes_ids())
                centers = np.array([base_graph.get_attributes_of_node(node_id)["center"] for node_id in node_ids])
                kdt = KDTree(centers, leaf_size=30, metric='euclidean')
                k = len(centers) if len(centers) <= settings["K_nearest_max"]+1 else settings["K_nearest_max"]+1
                query = kdt.query(centers, k=k, return_distance=False)
                query = np.array(list((map(lambda e: list(map(node_ids.__getitem__, e)), query))))
                base_nodes_ids = query[:, 0]
                all_target_nodes_ids = query[:, 1:]
                new_edges = []
                counter = 0
                for i, base_node_id in enumerate(base_nodes_ids):
                    target_nodes_ids = all_target_nodes_ids[i]
                    for target_node_id in target_nodes_ids:
                        tuple_direct, tuple_inverse = (base_node_id, target_node_id), (target_node_id, base_node_id)
                        distance = segments_distance(base_graph.get_attributes_of_node(base_node_id)["limits"],base_graph.get_attributes_of_node(target_node_id)["limits"])
                        rel_pos_1, _ = relative_positions(base_graph.get_attributes_of_node(base_node_id),base_graph.get_attributes_of_node(target_node_id))
                        def add_edge_features(feature_keys, feats):
                            if feature_keys[0] == "min_dist":
                                feats = np.concatenate([feats, distance]).astype(np.float32)  #, np.log(distance+1)]).astype(np.float32)
                            elif feature_keys[0] == "relative_pos":
                                feats = np.concatenate([feats, rel_pos_1[:2]]).astype(np.float32)
                            if len(feature_keys) > 1:
                                feats = add_edge_features(feature_keys[1:], feats)
                            return feats
                        x = add_edge_features(self.settings["initial_features"]["edge"], [])
                        if tuple_direct in positive_gt_edge_ids or tuple_inverse in positive_gt_edge_ids:
                            if not settings["use_gt"]:
                                new_edges.append((target_node_id, base_node_id,{"type": new_edge_types, "label": 1, "x":x, "viz_feat" : 'g', "linewidth":1.0, "alpha":0.5}))
                                # new_edges.append((target_node_id, base_node_id,{"type": new_edge_types, "label": 1, "x":x_2, "viz_feat" : 'g', "linewidth":1.0, "alpha":0.5}))
                                counter += 1
                            else:
                                new_edges.append((target_node_id, base_node_id,{"type": new_edge_types, "label": 0, "x":x, "viz_feat" : 'r', "linewidth":1.0, "alpha":0.5}))
                                counter += 1
                        else:
                            new_edges.append((target_node_id, base_node_id,{"type": new_edge_types, "label": 0, "x":x, "viz_feat" : 'r', "linewidth":1.0, "alpha":0.5}))
                            counter += 1
                            # new_edges.append((target_node_id, base_node_id,{"type": new_edge_types, "label": 0, "x":x_2, "viz_feat" : 'r', "linewidth":1.0, "alpha":0.5}))
                base_graph.unfreeze()
                base_graph.add_edges(new_edges)                

            ### Include random edges
            if settings["K_random_max"] > 0:
                nodes_ids = list(base_graph.filter_graph_by_node_types(settings["K_random_types"]).get_nodes_ids())
                for base_node_id in nodes_ids:
                    potential_nodes_ids = copy.deepcopy(nodes_ids)
                    potential_nodes_ids.remove(base_node_id)
                    random.shuffle(potential_nodes_ids)
                    random_nodes_ids = potential_nodes_ids[:settings["K_random_max"]]

                    new_edges = []
                    for target_node_id in random_nodes_ids:
                        tuple_direct = (base_node_id, target_node_id)
                        tuple_inverse = (tuple_direct[1], tuple_direct[0])
                        if tuple_direct not in list(base_graph.get_edges_ids()) and tuple_inverse not in list(base_graph.get_edges_ids()):
                            ### TODO Include X
                            new_edges.append((tuple_direct[0], tuple_direct[1],{"type": new_edge_types, "label": 0, "viz_feat" : 'blue', "linewidth":1.0, "alpha":0.5}))

                base_graph.unfreeze()
                base_graph.add_edges(new_edges)

            ### (un)direct
            if settings["directed"]:
                base_graph.to_directed()
            else:
                base_graph.to_undirected()

            base_graph.relabel_nodes(mapping = False, copy=True)
            new_nxdataset.append(base_graph)

        val_start_index = int(len(nxdataset)*(1-self.settings["training_split"]["val"]-self.settings["training_split"]["test"]))
        test_start_index = int(len(nxdataset)*(1-self.settings["training_split"]["test"]))
        extended_nxdatset = {"train" : new_nxdataset[:val_start_index], "val" : new_nxdataset[val_start_index:test_start_index],"test" : new_nxdataset[test_start_index:-1]}

        return extended_nxdatset


    def reintroduce_predicted_edges(self, unparented_base_graph, predictions, image_name = "name not provided"):
        unparented_base_graph = copy.deepcopy(unparented_base_graph)
        unparented_base_graph.add_edges(predictions)
        visualize_nxgraph(unparented_base_graph, image_name = image_name)

    
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
                        new_node_attrs["x"] = self.normalize_features("ws_node", node_attrs["x"])
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
                hdataset_key.append(from_networkxwrapper_2_heterodata(nxgraph))
            hdataset[key] = hdataset_key
        return hdataset
