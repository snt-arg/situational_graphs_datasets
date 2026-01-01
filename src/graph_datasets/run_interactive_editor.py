import queue
import pickle
import logging
from pathlib import Path

from graph_datasets.InteractiveGraphVisualizer import InteractiveGraphVisualizer as IGV
from graph_wrapper.GraphWrapper import GraphWrapper as GW

# config
GRAPH_DATASET_DIR = Path("~/workspaces/reasoning_ws/src/situational_graphs_datasets/datasets/ifh/viz").expanduser()
INTERACTIVE_DATASET_DIR = Path("~/workspaces/reasoning_ws/src/situational_graphs_datasets/datasets/ifh/viz").expanduser()
LOAD_INDEX = 0  # index to load specific file

# set logger
logger = logging.getLogger("vis")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

def load_and_sanitize_graph(idx: int) -> GW:
    """
    Loads a graph from a pkl file and re-wraps it to ensure GW class consistency

    Args:
        idx -> Int: Index of pkl file to load
    """
    # find pkl files
    pkl_files = sorted(GRAPH_DATASET_DIR.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl file found in {GRAPH_DATASET_DIR}")
    
    print("Available Files: ")
    for i, p in enumerate(pkl_files):
        print(f" [{i}] {p.name}")

    if idx >= len(pkl_files):
        print(f"Index {idx} out of range, loading last file")
        idx = -1

    chosen = pkl_files[idx]
    print(f"Loading Graph from {chosen}")

    # load pkl
    with open(chosen, "rb") as f:
        loaded_data = pickle.load(f)

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
    # load 
    graph = load_and_sanitize_graph(LOAD_INDEX)

    # debug
    print(f"Node Types in Graph: {graph.get_all_node_types()}")

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