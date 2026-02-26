import os
import pickle
from graph_wrapper.GraphWrapper import GraphWrapper
from graph_datasets.graph_visualizer import visualize_nxgraph, visualize_nxgraph_3d
import matplotlib.pyplot as plt



def main():
    root = "/media/adminpc/X9 Pro/Papers/ICML 2026/3DSG-Generation-Datasets_GraphWrapper"  # <- hardcode this
    out_root = os.path.join(root, "/media/adminpc/X9 Pro/Papers/ICML 2026/3DSG-Generation-Datasets_Networkx")
    os.makedirs(out_root, exist_ok=True)

    for name in os.listdir(root):
        if not name.endswith(".pkl"):
            continue

        pkl_path = os.path.join(root, name)
        with open(pkl_path, "rb") as f:
            graph_wrappers = pickle.load(f)

        folder_name = os.path.splitext(name)[0]
        graph_pickle_path = os.path.join(out_root, f"{folder_name}.pkl")

        networkx_graphs = []
        for idx, graph_wrapper in enumerate(graph_wrappers):
            print(f"dbg type(graph_wrapper) {type(graph_wrapper.graph)}")
            networkx_graphs.append(graph_wrapper.graph)
            visualize_nxgraph_3d(graph_wrapper, f"test", visualize_alone=True, include_node_ids=False, blocking=True)
            plt.show()
        print(len(networkx_graphs))
        print(type(networkx_graphs[0]))
        print(graph_pickle_path)
        with open(graph_pickle_path, "wb") as graph_file:
            pickle.dump(networkx_graphs, graph_file)

if __name__ == "__main__":
    main()
