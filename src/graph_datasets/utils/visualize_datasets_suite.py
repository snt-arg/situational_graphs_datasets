import os
import pickle
import random
import matplotlib.pyplot as plt
from graph_datasets.graph_visualizer import visualize_nxgraph_3d
from graph_wrapper.GraphWrapper import GraphWrapper
import copy
import numpy as np

root = "/media/adminpc/X9 Pro/Papers/ICML 2026/public_dataset"  # <- hardcode this
    
graphs_path = os.path.join(root, "graphs")
figures_path = os.path.join(root, "saved_figures")

def save_figure(fig, path):
    fig.savefig(path)
    plt.close(fig)


def process_method(method_name, nested_dict):

    os.makedirs(os.path.join(figures_path, method_name), exist_ok=True)

    for nested_dict_key in nested_dict.keys():
        out_path_subset = os.path.join(figures_path, method_name, nested_dict_key)
        os.makedirs(out_path_subset, exist_ok=True)
        for dataset_tag in nested_dict[nested_dict_key].keys():
            dataset_path = os.path.join(out_path_subset, dataset_tag)
            os.makedirs(dataset_path, exist_ok=True)
            
            graph_wrappers = nested_dict[nested_dict_key][dataset_tag]
            print(f"Total GraphWrappers: {len(graph_wrappers)}")
            
            # Randomly select 5 GraphWrappers
            selected_graphs = random.sample(graph_wrappers, min(5, len(graph_wrappers)))
            
            folder_name = os.path.splitext(dataset_tag)[0]
            folder_path = os.path.join(out_path_subset, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            # Visualize and save figures
            for idx, g in enumerate(selected_graphs):
                gw = GraphWrapper(graph_obj=g)

                # for node_attrs in gw.get_attributes_of_all_nodes():
                #     if node_attrs[1]["type"] == "building":
                #         node_attrs[1]["viz"]["feat"] = "cb"

                gw.remove_all_edges_between_node_types("ws","ws")
                for edge_attrs in gw.get_attributes_of_all_edges():
                    edge_attrs[2]["linewidth"] = 1.5
                    edge_attrs[2]["viz_feat"] = "a"
                
                # Visualize and tweak perspective
                fig = visualize_nxgraph_3d(
                    gw, 
                    f"graph_{idx}", 
                    visualize_alone=False, 
                    include_node_ids=False, 
                    hide_axes = True,
                    blocking=False,
                    zoom_factor = 0.5
                )

                # # Save the figure
                save_path = os.path.join(folder_path, f"graph_{idx}_3d.png")

                save_figure(fig, save_path)

                # print(f"Saved figure for graph {idx} to {save_path}")


def process_gt():
    GT_path = os.path.join(graphs_path, "GT")
    wg_nested_dict = {}
    for source_dataset_tag in os.listdir(GT_path):
        ws_dict = {}
        for pkl_path in os.listdir(os.path.join(GT_path, source_dataset_tag)):
            if not pkl_path.endswith(".pkl"):
                continue
            dataset_tag = pkl_path.split(".")[0]

            pkl_path = os.path.join(GT_path, source_dataset_tag, pkl_path)
            with open(pkl_path, "rb") as f:
                graph_wrappers = pickle.load(f)
            ws_dict[dataset_tag] = graph_wrappers
        wg_nested_dict[source_dataset_tag] = ws_dict
    process_method("GT",wg_nested_dict)


def process_reasonings():
    mapping = {"msd":"M-F", "ssg_manh_small":"S-F-square", "real":"R-F", "ssg_L_small":"S-F"}
    baselines_reasonings_path = os.path.join(graphs_path, "baselines_reasoning")
    wg_nested_dict = {"M":{"M-F":[]}, "R":{"R-F":[]}, "S":{"S-F-square":[], "S-F":[]}}
    method_mapping = {"s_graphs_1": "EC-square", "s_graphs_2": "EC-L"}

    for method_tag in method_mapping.keys():
        method_nested_dict = copy.deepcopy(wg_nested_dict)
        for pkl_file in os.listdir(baselines_reasonings_path):
            pkl_file_name = pkl_file.split(".")[0]
            with open(os.path.join(baselines_reasonings_path, pkl_file), "rb") as f:
                graph_wrappers = pickle.load(f)
            
            dataset_tag = mapping[pkl_file_name[:-3]]
            method_nested_dict[dataset_tag[0]][dataset_tag] += graph_wrappers[method_tag]

        process_method(method_mapping[method_tag], method_nested_dict)

def process_from_hpc():
    ifh_or_rand_path = os.path.join(graphs_path, "ifh_or_rand","networkx","ja")

    for pkl_file in os.listdir(ifh_or_rand_path):
        with open(os.path.join(ifh_or_rand_path, pkl_file), "rb") as f:
            graph_wrappers = pickle.load(f)
        for idx, g in enumerate(graph_wrappers):
            gw = GraphWrapper(graph_obj=g)
            # for node_attrs in gw.get_attributes_of_all_nodes():
            #     node_attrs[1]["center"] = node_attrs[1]["node_pos"]
            node_viz_feat_mapping = {
            'ws': "black",
            'room': 'ro',
            'wall': 'oo',
            'floor': 'go',
            'building': 'co',
            'wall_ws': 'yo',
            'city': 'ko'
            }
            for node_attrs in gw.get_attributes_of_all_nodes():
                if node_attrs[1]["type"] == "ws":
                    node_attrs[1]["limits"] = node_attrs[1]["viz"]["limits"]
                    node_attrs[1]["limits"][0][2] = node_attrs[1]["center"][2]
                    node_attrs[1]["limits"][1][2] = node_attrs[1]["center"][2]

                if node_attrs[1]["type"] == "city":
                    print(f'dbg city node_attrs[1]["center"] {node_attrs[1]["center"]}')

            viz_center_offsets = {"ws": np.array([0, 0, 0]).astype(np.float16), "room": np.array([0, 0, 2]).astype(np.float16), "wall": np.array([0, 0, 1]).astype(np.float16),\
                                   "floor": np.array([0, 0, 3]).astype(np.float16), "building": np.array([0, 0, 4]).astype(np.float16), "object": np.array([0, 0, 0.5]).astype(np.float16), "city": np.array([0, 0, 5]).astype(np.float16)}
            z_scaling = 1.
            for key in viz_center_offsets.keys():
                viz_center_offsets[key][2] = viz_center_offsets[key][2]*z_scaling

            gw._add_complete_viz_attributes_to_graph(viz_center_offsets, node_viz_feat_mapping)
            # fig = visualize_nxgraph_3d(
            #     gw, 
            #     f"graph_{idx}", 
            #     visualize_alone=True, 
            #     include_node_ids=False, 
            #     hide_axes = False,
            #     blocking=False,
            #     zoom_factor = 0.5
            # )
            # plt.show()

def process_ifh_rand():
    for method_name in ["ifh", "rand"]:
        print(f'processing {method_name}')

        method_path = os.path.join(graphs_path, method_name)
        wg_nested_dict = {}

        node_viz_feat_mapping = {
            'ws': "black",
            'room': 'ro',
            'wall': 'oo',
            'floor': 'go',
            'building': 'co',
            'wall_ws': 'yo',
            'city': 'ko'
        }
        viz_center_offsets = {"ws": np.array([0, 0, 0]).astype(np.float16), "room": np.array([0, 0, 2]).astype(np.float16),
                              "wall": np.array([0, 0, 1]).astype(np.float16), "floor": np.array([0, 0, 3]).astype(np.float16),
                              "building": np.array([0, 0, 4]).astype(np.float16), "object": np.array([0, 0, 0.5]).astype(np.float16),
                              "city": np.array([0, 0, 5]).astype(np.float16)}
        z_scaling = 0.2
        for key in viz_center_offsets.keys():
            viz_center_offsets[key][2] = viz_center_offsets[key][2]*z_scaling

        for dataset_tag in os.listdir(method_path):
            print(f'dbg processing {dataset_tag}')
            wg_nested_dict[dataset_tag] = {}
            for pkl_file in os.listdir(os.path.join(method_path, dataset_tag)):
                pkl_file_name = pkl_file.split(".")[0]
                with open(os.path.join(method_path, dataset_tag, pkl_file), "rb") as f:
                    graph_wrappers = pickle.load(f)

                for idx, g in enumerate(graph_wrappers):
                    gw = GraphWrapper(graph_obj=g)

                    for node_attrs in gw.get_attributes_of_all_nodes():
                        new_z = copy.deepcopy(node_attrs[1]["center"][2])
                        if node_attrs[1]["type"] == "ws":
                            node_attrs[1]["limits"] = copy.deepcopy(node_attrs[1]["viz"]["limits"])
                            # node_attrs[1]["limits"][0][2] = new_z
                            # node_attrs[1]["limits"][1][2] = new_z
                            # node_attrs[1]["limits"] = copy.deepcopy(node_attrs[1]["viz"]["limits"])

                    gw._add_complete_viz_attributes_to_graph(viz_center_offsets, node_viz_feat_mapping)
                
                wg_nested_dict[dataset_tag][pkl_file_name] = graph_wrappers

            process_method(method_name, wg_nested_dict)
    

def main():
    
    # process_gt()
    # process_reasonings()
    # process_from_hpc()
    process_ifh_rand()


if __name__ == "__main__":
    main()
