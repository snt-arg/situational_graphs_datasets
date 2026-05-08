import queue
import pickle
import logging
from pathlib import Path
import joblib

from graph_datasets.InteractiveGraphVisualizer import InteractiveGraphVisualizer as IGV
from graph_wrapper.GraphWrapper import GraphWrapper as GW


class ModuleRemappingUnpickler(pickle.Unpickler):
    """Custom unpickler that remaps old module names to new ones"""
    
    def find_class(self, module, name):
        # Remap module name: situational_graphs_wrapper -> graph_wrapper
        if module.startswith("situational_graphs_wrapper"):
            module = module.replace("situational_graphs_wrapper", "graph_wrapper", 1)
        return super().find_class(module, name)

# config
INTERACTIVE_DATASET_DIR = Path("~/workspaces/reasoning_ws/src/situational_graphs_datasets/datasets/nimrod/topfloor_edge_debug ").expanduser()
OUTPUT_DIR = INTERACTIVE_DATASET_DIR
LOAD_INDEX = 0  # index to load specific file

# set logger
logger = logging.getLogger("vis")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

def load_and_sanitize_graph(idx: int = None) -> GW:
    """
    Loads a graph from a pkl file and re-wraps it to ensure GW class consistency

    Args:
        idx -> Int: Index of pkl file to load. If None, user will be prompted to select.
    """
    selected = OUTPUT_DIR

    # find pkl files
    pkl_files = sorted(selected.glob("*.pkl")) + sorted(selected.glob("*.joblib"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl file found in {selected}")
        
    print("Available Files: ")
    for i, p in enumerate(pkl_files):
        print(f" [{i}] {p.name}")

    # If idx not provided, ask user to select
    if idx is None:
        while True:
            try:
                idx = int(input(f"\nSelect file index (0-{len(pkl_files)-1}): "))
                if 0 <= idx < len(pkl_files):
                    break
                else:
                    print(f"Invalid index. Please enter a number between 0 and {len(pkl_files)-1}")
            except ValueError:
                print("Invalid input. Please enter a number.")

    if idx >= len(pkl_files):
        print(f"Index {idx} out of range, loading last file")
        idx = -1

    chosen = pkl_files[idx]
    print(f"Loading Graph from {chosen}")

    # load pkl
    if chosen.suffix == ".joblib":
        loaded_data = joblib.load(chosen)

    else:
        with open(chosen, "rb") as f:
            loaded_data = ModuleRemappingUnpickler(f).load()

    # handle sdg data
    raw_graph_obj = None

    if isinstance(loaded_data, list) and len(loaded_data) > 0:
        print(f"Dataset is a list of {len(loaded_data)} graphs. Loading first graph.")
        raw_graph_obj = loaded_data[0]
    else:
        raw_graph_obj = loaded_data

    # re-wrap into GW
    try:
        # assuming raw is GW
        clean_gw = GW(graph_obj=raw_graph_obj.graph)
        print(f"Graph loaded successfully. Nodes: {raw_graph_obj.get_total_number_nodes()}")
        return raw_graph_obj
    except AttributeError:
        # fallback if raw is not GW
        clean_gw = GW(graph_obj=raw_graph_obj)
        return clean_gw
    

def main():
    # load - prompts user to select file from available pkl files
    graph = load_and_sanitize_graph()

    # debug
    # print(f"Node Types in Graph: {graph.get_all_node_types()}")

    # prep queue
    grp_q = queue.Queue()
    grp_update_q = queue.Queue()

    # init IGV
    viz = IGV(
        graph=graph,
        image_name=f"Interactive Editor - {graph.name if hasattr(graph, 'name') else 'Graph'}",
        group_queue=grp_q,
        graph_update_queue=grp_update_q,
        callback=None,
        logger=logger,
        full_graph=graph,
        default_save_dir=INTERACTIVE_DATASET_DIR
    )

    # run
    viz.show()

if __name__ == "__main__":
    main()