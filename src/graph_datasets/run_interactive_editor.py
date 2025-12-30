import queue
import pickle
import logging
from pathlib import Path

from graph_datasets.InteractiveGraphVisualizer import InteractiveGraphVisualizer as IGV
from graph_wrapper.GraphWrapper import GraphWrapper as GW

# config
GRAPH_DATASET_DIR = Path("/home/sven/project/Dataset/Synthetic")
INTERACTIVE_DATASET_DIR = Path("/home/sven/project/Dataset/Interactive")
OUTPUT_DIR = Path("/home/sven/project/situational_graphs_datasets/src/graph_datasets/output_dataset")
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
    selected = OUTPUT_DIR

    # find pkl files
    pkl_files = sorted(selected.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl file found in {selected}")
    
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

    return raw_graph_obj  # type: ignore

def main():
    # load 
    graph = load_and_sanitize_graph(LOAD_INDEX)

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