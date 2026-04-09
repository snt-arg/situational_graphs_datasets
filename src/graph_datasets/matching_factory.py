import random
import os

from graph_visualizer import visualize_nxgraph_3d
from graph_wrapper.GraphWrapper import GraphWrapper
from graph_datasets.DatasetFactory import DatasetFactory

TARGET_LAYOUTS = 100
MIN_ROOMS_REQUIRED = 1
MAX_ROOMS_REQUIRED = 6
MAX_ADAPTIVE_ATTEMPTS = 10
CANDIDATE_MULTIPLIER = 6
CONFIG_NAME = "matching/incremental_local_symmetries"

visualize = False
visualize_s_graph_list = False
save_pickle = True
save_pickle_path = "/home/adminpc/workspaces/reasoning_ws/src/situational_graphs_datasets/datasets/matching/"
pickle_name = "incremental_local_symmetries_max6rooms.pkl"


def count_room_nodes(graph):
    return sum(
        1
        for _, node_attrs in graph.get_attributes_of_all_nodes()
        if node_attrs.get("type") == "room"
    )


def build_matching_dataset(n_graphs_reduction):
    factory = DatasetFactory(
        dataset_base="synthetic",
        config_name=CONFIG_NAME,
        extension_name="original",
        pickle_name=pickle_name,
        n_graphs_reduction=n_graphs_reduction,
        save_pickle_path=save_pickle_path,
    )

    # Preserve original flow: create dataset first, then compose matching pairs.
    factory.create_dataset()
    factory.matching_extension()
    return factory, factory.all_dataset


def select_exact_100_with_coverage(dataset, room_counts, target_layouts, min_rooms, max_rooms):
    valid_idxs = [i for i, c in enumerate(room_counts) if min_rooms <= c <= max_rooms]
    idxs_min = [i for i in valid_idxs if room_counts[i] == min_rooms]
    idxs_max = [i for i in valid_idxs if room_counts[i] == max_rooms]

    if len(valid_idxs) < target_layouts or not idxs_min or not idxs_max:
        return None, None

    selected = []
    used = set()

    # Force presence of required extremes.
    selected.append(idxs_min[0])
    used.add(idxs_min[0])

    if idxs_max[0] not in used:
        selected.append(idxs_max[0])
        used.add(idxs_max[0])

    remaining = [i for i in valid_idxs if i not in used]
    random.shuffle(remaining)

    selected.extend(remaining[: target_layouts - len(selected)])
    selected = selected[:target_layouts]

    selected_dataset = [dataset[i] for i in selected]
    selected_counts = [room_counts[i] for i in selected]

    return selected_dataset, selected_counts


def generate_dataset_with_coverage(target_layouts, min_rooms, max_rooms, max_attempts):
    random.seed(7)

    for attempt in range(1, max_attempts + 1):
        candidate_layouts = target_layouts * CANDIDATE_MULTIPLIER
        print(f"\nAdaptive attempt {attempt}/{max_attempts}")
        print(f"Using config_name={CONFIG_NAME}, n_graphs_reduction={candidate_layouts}")

        factory, composed_dataset = build_matching_dataset(candidate_layouts)

        room_counts = []
        for a_graph, _s_graphs in composed_dataset:
            if isinstance(a_graph, GraphWrapper):
                room_counts.append(count_room_nodes(a_graph))

        if not room_counts:
            raise RuntimeError("No layouts were generated.")

        obs_min = min(room_counts)
        obs_max = max(room_counts)
        print(f"Candidate pool stats -> size: {len(room_counts)}, min: {obs_min}, max: {obs_max}")

        selected_dataset, selected_counts = select_exact_100_with_coverage(
            composed_dataset,
            room_counts,
            target_layouts,
            min_rooms,
            max_rooms,
        )

        if selected_dataset is not None:
            selected_min = min(selected_counts)
            selected_max = max(selected_counts)
            all_within = all(min_rooms <= c <= max_rooms for c in selected_counts)

            # Enforce exact requested extrema on the FINAL dataset.
            if (
                len(selected_counts) == target_layouts
                and all_within
                and selected_min == min_rooms
                and selected_max == max_rooms
            ):
                print(
                    "Coverage satisfied on selected dataset -> "
                    f"size: {len(selected_counts)}, min: {selected_min}, max: {selected_max}"
                )
                return factory, selected_dataset, selected_counts

            print(
                "Selected dataset did not meet exact extrema constraint; retrying. "
                f"selected_min={selected_min}, selected_max={selected_max}"
            )

    raise RuntimeError(
        "Unable to satisfy 1..8 room coverage in 100 layouts with adaptive tuning. "
        "Try increasing CANDIDATE_MULTIPLIER or MAX_ADAPTIVE_ATTEMPTS."
    )


dataset_factory, all_dataset, room_counts = generate_dataset_with_coverage(
    target_layouts=TARGET_LAYOUTS,
    min_rooms=MIN_ROOMS_REQUIRED,
    max_rooms=MAX_ROOMS_REQUIRED,
    max_attempts=MAX_ADAPTIVE_ATTEMPTS,
)


if visualize:
    for a_graph, s_graph_list in all_dataset:
        if isinstance(a_graph, GraphWrapper):
            visualize_nxgraph_3d(
                a_graph,
                "Agraph",
                visualize_alone=True,
                include_node_ids=False,
                blocking=True,
                show_hover_tooltips=True,
                hide_axes=True,
            )

        if visualize_s_graph_list and isinstance(s_graph_list, list):
            for i, s_graph in enumerate(s_graph_list):
                if isinstance(s_graph, GraphWrapper):
                    # Block only on the last view to keep browsing convenient.
                    is_last = i == len(s_graph_list) - 1
                    visualize_nxgraph_3d(
                        s_graph,
                        f"Sgraph {i}",
                        visualize_alone=True,
                        include_node_ids=False,
                        blocking=is_last,
                        show_hover_tooltips=True,
                        hide_axes=True,
                    )


if room_counts:
    unique_counts = sorted(set(room_counts))
    print("\n=== Layout Room Count Analysis ===")
    print(f"Layouts analyzed: {len(room_counts)}")
    print(f"Minimum room nodes in a layout: {min(room_counts)}")
    print(f"Maximum room nodes in a layout: {max(room_counts)}")
    print(f"Unique room-count values observed: {unique_counts}")
    print(f"Contains {MIN_ROOMS_REQUIRED} room layout: {MIN_ROOMS_REQUIRED in room_counts}")
    print(f"Contains {MAX_ROOMS_REQUIRED} room layout: {MAX_ROOMS_REQUIRED in room_counts}")
    print(
        "All layouts inside required range "
        f"[{MIN_ROOMS_REQUIRED}, {MAX_ROOMS_REQUIRED}]: "
        f"{all(MIN_ROOMS_REQUIRED <= c <= MAX_ROOMS_REQUIRED for c in room_counts)}"
    )


if save_pickle:
    full_save_path = os.path.join(save_pickle_path, pickle_name)
    os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
    dataset_factory.save_dataset(dataset=all_dataset, save_path=full_save_path)
    print(f"Saved dataset to {full_save_path}")
