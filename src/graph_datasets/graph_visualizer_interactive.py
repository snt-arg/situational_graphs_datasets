import matplotlib.pyplot as plt
import numpy as np

# Global dictionaries to store groups.
# active_groups maps group type -> active group (set of node IDs)
# finalized_groups maps group type -> list of finalized groups (each a set of node IDs)
active_groups = {}
finalized_groups = {}
current_group_type = "R"  # Default current group type now set to "R"

# Define colors for group types.
group_colors = {
    "R": "red",
    "r": "orange",
    "W": "brown",
    "w": "black"
}

def visualize_nxgraph_interactive(graph, image_name, visualize_alone=True, include_node_ids=True, logger=None):
    global active_groups, finalized_groups, current_group_type
    active_groups = {}
    finalized_groups = {}
    current_group_type = "R"  # Default group type
    
    fig = plt.figure(image_name)
    ax = fig.add_subplot(111)
    
    # Dictionary to store node id -> 2D coordinate for selection.
    node_coords = {}
    
    # Draw nodes (and extra drawings for "Line" type nodes).
    nodes_data = graph.get_attributes_of_all_nodes()  # list of (node_id, attr)
    for node_data in nodes_data:
        node_id, attr = node_data[0], node_data[1]
        if attr["viz_type"] == "Point":
            coords = np.array(attr["viz_data"])[:2]
        elif attr["viz_type"] == "Line":
            # For Line nodes, use the provided center for selection.
            coords = np.array(attr["center"])[:2]
            # Draw the main line (using viz_data)
            viz_data = np.array(attr["viz_data"])[:, :2]
            linewidth = attr.get("linewidth", 1.5)
            ax.plot(viz_data[:,0], viz_data[:,1], attr["viz_feat"], linewidth=linewidth)
            # Draw the normal indicator (blue) from center to center + normal/4.
            center = np.array(attr["center"])[:2]
            normal = np.array(attr["normal"])[:2]
            norm_line = np.stack([center, center + normal/4])
            ax.plot(norm_line[:,0], norm_line[:,1], "b", linewidth=linewidth)
        else:
            continue

        # Draw node marker.
        ax.plot(coords[0], coords[1], attr["viz_feat"])
        node_coords[node_id] = coords
        if include_node_ids:
            plt.text(coords[0], coords[1], str(node_id), fontsize=12, color='black')
    
    # Draw edges using nodes' center coordinates.
    edges_data = graph.get_attributes_of_all_edges()  # list of (node1, node2, attr)
    for edge_data in edges_data:
        node_id1, node_id2, attr = edge_data
        p1 = np.array(graph.get_attributes_of_node(node_id1)["center"])[:2]
        p2 = np.array(graph.get_attributes_of_node(node_id2)["center"])[:2]
        points = np.vstack([p1, p2])
        viz_feat = attr.get("viz_feat", "")
        linewidth = attr.get("linewidth", 1.5)
        alpha = attr.get("alpha", 1.0)
        ax.plot(points[:,0], points[:,1], viz_feat, linewidth=linewidth, alpha=alpha)
        if "pred" in attr:
            center_x = (points[0,0] + points[1,0]) / 2
            center_y = (points[0,1] + points[1,1]) / 2
            ax.text(center_x, center_y, "{:.2f}".format(attr['pred']))
    
    ax.set_aspect('equal', adjustable='datalim')
    
    # Function to update selection overlay.
    def update_selection():
        # Remove previous selection overlays (identified by label 'group_sel').
        for artist in ax.get_children():
            if hasattr(artist, 'get_label') and artist.get_label() == 'group_sel':
                artist.remove()
        # Overlay markers for each active group.
        for grp_type, group in active_groups.items():
            sel_x = []
            sel_y = []
            for node_id in group:
                if node_id in node_coords:
                    sel_x.append(node_coords[node_id][0])
                    sel_y.append(node_coords[node_id][1])
            if sel_x and sel_y:
                ax.scatter(sel_x, sel_y, s=150, facecolors='none', 
                           edgecolors=group_colors.get(grp_type, "red"),
                           linewidths=2, label='group_sel')
    
    # Mouse click event handler.
    def on_click(event):
        global active_groups, current_group_type
        if event.inaxes != ax:
            return
        click_point = np.array([event.xdata, event.ydata])
        tolerance = 0.5  # Adjust tolerance as needed.
        hit = False
        for node_id, coords in node_coords.items():
            if np.linalg.norm(click_point - coords) < tolerance:
                if current_group_type not in active_groups:
                    active_groups[current_group_type] = set()
                if node_id in active_groups[current_group_type]:
                    active_groups[current_group_type].remove(node_id)
                    if logger:
                        logger.info(f"Node {node_id} removed from group '{current_group_type}'.")
                else:
                    active_groups[current_group_type].add(node_id)
                    if logger:
                        logger.info(f"Node {node_id} added to group '{current_group_type}'.")
                hit = True
                update_selection()
                fig.canvas.draw_idle()
                break
        if not hit:
            if logger:
                logger.info("Background click detected; use Enter to finalize the current group.")
    
    # Key press event handler.
    def on_key(event):
        global current_group_type, active_groups, finalized_groups
        if event.key in group_colors:
            current_group_type = event.key
            if logger:
                logger.info(f"Current group type set to '{current_group_type}'.")
        elif event.key == "enter":
            if current_group_type in active_groups and active_groups[current_group_type]:
                if current_group_type not in finalized_groups:
                    finalized_groups[current_group_type] = []
                finalized_groups[current_group_type].append(active_groups[current_group_type].copy())
                if logger:
                    logger.info(f"Finalized group '{current_group_type}': {active_groups[current_group_type]}")
                active_groups[current_group_type] = set()
                update_selection()
                fig.canvas.draw_idle()
            else:
                if logger:
                    logger.info("No nodes selected in current group to finalize.")
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.title(f"Interactive Graph - Current group type: {current_group_type}")
    
    if visualize_alone:
        plt.show()
    else:
        plt.close(fig)
    
    # Return the finalized groups dictionary.
    return finalized_groups

# -----------------------------------------
# Example usage:
# Assume your graph object implements:
#   get_attributes_of_all_nodes(), get_attributes_of_node(node_id), get_attributes_of_all_edges()
#
# from your_graph_module import your_graph
# groups = visualize_nxgraph_interactive(your_graph, "InteractiveGraph", visualize_alone=True, logger=your_logger)
# Now, 'groups' is a dictionary mapping group types (e.g. "R", "r", "W", "w") to a list of finalized groups.
