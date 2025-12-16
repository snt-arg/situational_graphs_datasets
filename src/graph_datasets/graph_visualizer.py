import matplotlib
import os

matplotlib.use("TkAgg")   # or "Qt5Agg"

import matplotlib.pyplot as plt


import numpy as np
import networkx as nx
import torch
# from matplotlib.patches import ConnectionPatch
from mpl_toolkits.mplot3d import proj3d  # Add this import at the top of your file



def visualize_nxgraph(graph, image_name, visualize_alone=False, include_node_ids=True, logger = None):
    nodes_data = graph.get_attributes_of_all_nodes()
    fig = plt.figure(image_name)
    # fig.clf()
    ax = fig.add_subplot(111)
    for node_data in nodes_data:
        if node_data[1]["viz"]["type"] == "Point":
            if "markersize" in node_data[1].keys():
                markersize = node_data[1]["markersize"]
            else:
                markersize = 1.0
            ax.plot(node_data[1]["viz"]["center"][0], node_data[1]["viz"]["center"][1], node_data[1]["viz"]["center"], markersize=markersize*10)
            tag_center = node_data[1]["viz"]["center"]

        elif node_data[1]["viz"]["type"] == "Line":
            viz_data = np.array(node_data[1]["viz"]["limits"])
            linewidth = node_data[1]["viz"]["linewidth"] if "linewidth" in node_data[1]["viz"].keys() else 1.5

            ax.plot(viz_data[:,0], viz_data[:,1], node_data[1]["viz"]["feat"], linewidth=linewidth)

            norm_line = np.stack([node_data[1]["center"], node_data[1]["center"] + node_data[1]["normal"]/4])
            ax.plot(norm_line[:,0], norm_line[:,1], "b", linewidth=linewidth)
            tag_center = np.array(node_data[1]["center"]) + np.array(node_data[1]["normal"]) * 0.5

        if include_node_ids:
            plt.text(tag_center[0], tag_center[1], str(node_data[0]), fontsize=12, color='black')

    edges_data = graph.get_attributes_of_all_edges()
    for edge_data in edges_data:
        points = np.array([nodes_data[edge_data[0]]["viz"]["center"], nodes_data[edge_data[1]]["viz"]["center"]])
        viz_feat = edge_data[2]["viz_feat"] if "viz_feat" in edge_data[2].keys() else ""
        linewidth = edge_data[2]["linewidth"] if "linewidth" in edge_data[2].keys() else 1.5
        alpha = edge_data[2]["alpha"] if "alpha" in edge_data[2].keys() else 1.0
        ax.plot(points[:,0], points[:,1], viz_feat, linewidth=linewidth, alpha=alpha)

        if "pred" in list(edge_data[2].keys()):
            center_x, center_y = (points[0,0] + points[1,0]) / 2, (points[0,1] + points[1,1]) / 2
            ax.text(center_x, center_y, "{:.2f}".format(edge_data[2]['pred']))
    # plt.xlim([-3, 23])
    # plt.ylim([-3, 23])

    # plt.draw()
    # plt.pause(0.001)
    # plt.show()
    ax.set_aspect('equal', adjustable='datalim')
    # ax.autoscale()

    # plt.tight_layout()
    if visualize_alone:
        # ax.draw()
        # ax.pause(0.001)
        plt.show()

    else:
        plt.close(fig)
    return fig

def visualize_nxgraph_3d(graph, image_name, visualize_alone=False, include_node_ids=True, logger=None, blocking=False):
    nodes_data = graph.get_attributes_of_all_nodes()
    node_attr_dict = {nd[0]: nd[1] for nd in nodes_data}
    fig = plt.figure(image_name)
    ax = fig.add_subplot(111, projection='3d')

    def to_3d(arr):
        arr = np.array(arr)
        if arr.shape[-1] == 2:
            arr = np.append(arr, 0)
        return arr

    node_positions = {}
    node_artists = []
    plane_artists = []  # (node_id, normal_artist, center, normal, main_line_artist)
    edge_artists = []
    edges_data = graph.get_attributes_of_all_edges()

    legend_handles = {}
    for node_data in nodes_data:
        node_id = node_data[0]
        if node_data[1]["viz"]["type"] == "Point":
            markersize = node_data[1].get("markersize", 1.0)
            viz_data = to_3d(node_data[1]["viz"]["center"])
            color = _mpl_color_from_feat(node_data[1]["viz"]["feat"])
            marker = node_data[1]["viz"]["feat"][1] if len(node_data[1]["viz"]["feat"]) > 1 else 'o'
            label = node_data[1].get("type", "Point")
            if label not in legend_handles:
                h = ax.scatter([], [], [], marker=marker, s=markersize*30, color=color, label=label)
                legend_handles[label] = h
            artist = ax.scatter(viz_data[0], viz_data[1], viz_data[2], marker=marker, s=markersize*30, color=color, picker=True)
            node_artists.append((node_id, artist))
            node_positions[node_id] = viz_data
            tag_center = viz_data
        elif node_data[1]["viz"]["type"] == "Line":
            viz_data = np.array(node_data[1]["viz"]["limits"])
            linewidth = node_data[1]["viz"].get("linewidth", 1.5)
            color = _mpl_color_from_feat(node_data[1]["viz"]["feat"])
            label = node_data[1]["viz"].get("type", "Line")
            if viz_data.shape[1] == 2:
                viz_data = np.hstack([viz_data, np.zeros((viz_data.shape[0], 1))])
            if label not in legend_handles:
                h, = ax.plot([], [], [], color=color, linewidth=linewidth, label=label)
                legend_handles[label] = h
            # Main line
            main_line_artist, = ax.plot(viz_data[:,0], viz_data[:,1], viz_data[:,2], color=color, linewidth=linewidth)
            # Blue normal line (plane node)
            center = to_3d(node_data[1]["center"])
            normal = to_3d(node_data[1].get("normal", [0, 0, 0]))
            norm_line = np.stack([center, center + normal/4])
            normal_artist, = ax.plot(norm_line[:,0], norm_line[:,1], norm_line[:,2], color='b', linewidth=linewidth)
            plane_artists.append((node_id, normal_artist, center, normal, main_line_artist))  # Track plane node
            tag_center = center + normal * 0.5 if np.linalg.norm(normal) > 0 else center
        if include_node_ids:
            ax.text(tag_center[0], tag_center[1], tag_center[2], str(node_data[0]), fontsize=10, color='black')


    # Draw edges and store artists
    for edge_data in edges_data:
        points = np.array([to_3d(nodes_data[edge_data[0]]["viz"]["center"]), to_3d(nodes_data[edge_data[1]]["viz"]["center"])])
        color = _mpl_color_from_feat(edge_data[2].get("viz_feat", "k"))
        linewidth = edge_data[2].get("linewidth", 1.5)
        alpha = edge_data[2].get("alpha", 1.0)
        label = edge_data[2].get("type", "Edge")
        if label not in legend_handles:
            h, = ax.plot([], [], [], color=color, linewidth=linewidth, alpha=alpha, label=label)
            legend_handles[label] = h
        artist, = ax.plot(points[:,0], points[:,1], points[:,2], color=color, linewidth=linewidth, alpha=alpha)
        edge_artists.append((edge_data, artist))
        if "pred" in edge_data[2]:
            center = (points[0] + points[1]) / 2
            ax.text(center[0], center[1], center[2], "{:.2f}".format(edge_data[2]['pred']), fontsize=9)
    ax.set_box_aspect([1,1,1])

    # Calculate the bounds including all elements
    x_coords = []
    y_coords = []
    z_coords = []

    # Collect coordinates from all visual elements
    for node_data in nodes_data:
        if node_data[1]["viz"]["type"] == "Point":
            pos = to_3d(node_data[1]["viz"]["center"])
            x_coords.append(pos[0])
            y_coords.append(pos[1])
            z_coords.append(pos[2])
        elif node_data[1]["viz"]["type"] == "Line":
            # Get line endpoints
            line_data = np.array(node_data[1]["viz"]["limits"])
            if line_data.shape[1] == 2:
                line_data = np.hstack([line_data, np.zeros((line_data.shape[0], 1))])
            x_coords.extend(line_data[:, 0])
            y_coords.extend(line_data[:, 1])
            z_coords.extend(line_data[:, 2])
            # Get center and normal
            center = to_3d(node_data[1]["center"])
            normal = to_3d(node_data[1].get("normal", [0, 0, 0]))
            endpoint = center + normal/4
            x_coords.extend([center[0], endpoint[0]])
            y_coords.extend([center[1], endpoint[1]])
            z_coords.extend([center[2], endpoint[2]])

    # Include edge endpoints and labels
    for edge_data in edges_data:
        source = to_3d(nodes_data[edge_data[0]]["viz"]["center"])
        target = to_3d(nodes_data[edge_data[1]]["viz"]["center"])
        x_coords.extend([source[0], target[0]])
        y_coords.extend([source[1], target[1]])
        z_coords.extend([source[2], target[2]])
        if "pred" in edge_data[2]:
            # Include label position
            mid = (source + target) / 2
            x_coords.append(mid[0])
            y_coords.append(mid[1])
            z_coords.append(mid[2])

    if x_coords:  # Only proceed if we have coordinates
        # Calculate the ranges and add padding for each dimension
        def get_padded_range(coords):
            if not coords:
                return -1, 1  # Default range if no coordinates
            min_val = np.min(coords)
            max_val = np.max(coords)
            span = max_val - min_val
            if span == 0:  # Handle single point case
                span = 1.0
            padding = span * 0.2  # Exactly 20% padding of the dimension's span
            return min_val - padding, max_val + padding

        # Get padded ranges for each dimension
        x_min, x_max = get_padded_range(x_coords)
        y_min, y_max = get_padded_range(y_coords)
        z_min, z_max = get_padded_range(z_coords)
        
        # Set the limits independently for each dimension
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

    ax.legend()

    # --- Interactivity: highlight node, plane node, and edges on hover ---
    def on_motion(event):
        if event.inaxes != ax:
            return
        min_dist = float('inf')
        closest_node = None
        closest_plane = None

        # Check point nodes
        for node_id, pos in node_positions.items():
            x2, y2, _ = proj3d.proj_transform(pos[0], pos[1], pos[2], ax.get_proj())
            dist = np.hypot(event.x - ax.transData.transform((x2, y2))[0], event.y - ax.transData.transform((x2, y2))[1])
            if dist < min_dist and dist < 30:
                min_dist = dist
                closest_node = node_id
                closest_plane = None

        # Check plane nodes (blue lines)
        for node_id, normal_artist, center, normal, main_line_artist in plane_artists:
            plane_mid = center + normal * 0.5
            x2, y2, _ = proj3d.proj_transform(plane_mid[0], plane_mid[1], plane_mid[2], ax.get_proj())
            dist = np.hypot(event.x - ax.transData.transform((x2, y2))[0], event.y - ax.transData.transform((x2, y2))[1])
            if dist < min_dist and dist < 30:
                min_dist = dist
                closest_plane = node_id
                closest_node = None

        # Reset all nodes/edges/planes
        for nid, artist in node_artists:
            artist.set_facecolor(_mpl_color_from_feat(node_attr_dict[nid]["viz"]["feat"]))
            artist.set_sizes([node_attr_dict[nid].get("markersize", 1.0)*30])
        for edge_data, artist in edge_artists:
            artist.set_color(_mpl_color_from_feat(edge_data[2].get("viz_feat", "k")))
            artist.set_linewidth(edge_data[2].get("linewidth", 1.5))
        for nid, normal_artist, center, normal, main_line_artist in plane_artists:
            normal_artist.set_color('b')
            normal_artist.set_linewidth(node_attr_dict[nid]["viz"].get("linewidth", 1.5))
            main_line_artist.set_color(_mpl_color_from_feat(node_attr_dict[nid]["viz"]["feat"]))
            main_line_artist.set_linewidth(node_attr_dict[nid]["viz"].get("linewidth", 1.5))

        # Highlight if found
        connected_nodes = set()
        if closest_node is not None:
            for nid, artist in node_artists:
                if nid == closest_node:
                    artist.set_facecolor('yellow')
                    artist.set_sizes([80])
            for edge_data, artist in edge_artists:
                if edge_data[0] == closest_node or edge_data[1] == closest_node:
                    artist.set_color('orange')
                    artist.set_linewidth(3)
                    connected_nodes.add(edge_data[0])
                    connected_nodes.add(edge_data[1])
            for nid, artist in node_artists:
                if nid in connected_nodes and nid != closest_node:
                    artist.set_facecolor('orange')
                    artist.set_sizes([60])
            # Highlight plane node neighbors
            for nid, normal_artist, center, normal, main_line_artist in plane_artists:
                if nid in connected_nodes and nid != closest_node:
                    normal_artist.set_color('orange')
                    normal_artist.set_linewidth(4)
                    main_line_artist.set_color('orange')
                    main_line_artist.set_linewidth(4)
        elif closest_plane is not None:
            for nid, normal_artist, center, normal, main_line_artist in plane_artists:
                if nid == closest_plane:
                    normal_artist.set_color('orange')
                    normal_artist.set_linewidth(4)
                    main_line_artist.set_color('orange')
                    main_line_artist.set_linewidth(4)
            for edge_data, artist in edge_artists:
                if edge_data[0] == closest_plane or edge_data[1] == closest_plane:
                    artist.set_color('orange')
                    artist.set_linewidth(3)
                    connected_nodes.add(edge_data[0])
                    connected_nodes.add(edge_data[1])
            for nid, artist in node_artists:
                if nid in connected_nodes:
                    artist.set_facecolor('orange')
                    artist.set_sizes([60])
            # Highlight plane node neighbors
            for nid, normal_artist, center, normal, main_line_artist in plane_artists:
                if nid in connected_nodes and nid != closest_plane:
                    normal_artist.set_color('orange')
                    normal_artist.set_linewidth(4)
                    main_line_artist.set_color('orange')
                    main_line_artist.set_linewidth(4)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    if visualize_alone:
        plt.show(block=blocking)
    else:
        plt.close(fig)
    return fig


def _mpl_color_from_feat(viz_feat):
    """Helper to extract matplotlib color from viz_feat string like 'go', 'ro', etc."""
    if isinstance(viz_feat, str) and len(viz_feat) > 0:
        color_dict = {
            'g': 'green',
            'r': 'red',
            'b': 'blue',
            'k': 'black',
            'c': 'cyan',
            'm': 'magenta',
            'y': 'yellow',
            'o': 'orange'
        }
        c = viz_feat[0]
        return color_dict.get(c, c)
    return 'k'

def visualize_nxgraph_pair(graph1, graph2, image_name, visualize_alone=False, g1digraph=False, g2digraph=False):
    fig = plt.figure(image_name)
    fig.clf()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    # get nodes and edges data
    if g1digraph:
        nodes_data1 = graph1.nodes(data=True)
        edges_data1 = graph1.edges(data=True)
    else:
        nodes_data1 = graph1.get_attributes_of_all_nodes()
        edges_data1 = graph1.get_attributes_of_all_edges()
    if g2digraph:
        nodes_data2 = graph2.nodes(data=True)
        edges_data2 = graph2.edges(data=True)
    else:
        nodes_data2 = graph2.get_attributes_of_all_nodes()
        edges_data2 = graph2.get_attributes_of_all_edges()
          
    for node_data in nodes_data1:
        if node_data[1]["viz"]["type"] == "Point":
            ax1.plot(node_data[1]["viz"]["center"][0], node_data[1]["viz"]["center"][1], node_data[1]["viz"]["feat"])
        elif node_data[1]["viz"]["type"] == "Line":
            viz_data = np.array(node_data[1]["viz"]["limits"])
            ax1.plot(viz_data[:,0], viz_data[:,1], node_data[1]["viz"]["feat"])
    for node_data in nodes_data2:
        if node_data[1]["viz"]["type"] == "Point":
            ax2.plot(node_data[1]["viz"]["center"][0], node_data[1]["viz"]["center"][1], node_data[1]["viz"]["feat"])
        elif node_data[1]["viz"]["type"] == "Line":
            viz_data = np.array(node_data[1]["viz"]["limits"])
            ax2.plot(viz_data[:,0], viz_data[:,1], node_data[1]["viz"]["feat"])

    for edge_data in edges_data1:
        points = np.array([nodes_data1[edge_data[0]]["center"], nodes_data1[edge_data[1]]["center"]])
        ax1.plot(points[:,0], points[:,1], edge_data[2]["viz"]["feat"])
    for edge_data in edges_data2:
        points = np.array([nodes_data2[edge_data[0]]["center"], nodes_data2[edge_data[1]]["center"]])
        ax2.plot(points[:,0], points[:,1], edge_data[2]["viz"]["feat"])

    ax1.set_aspect('equal', adjustable='datalim')
    ax2.set_aspect('equal', adjustable='datalim')
    
    plt.draw()
    
    if visualize_alone:
        plt.show()
    else:
        plt.close(fig)
    
    return fig

def plot_matching_with_visualization(graph1, graph2, match_result, image_name="Graph Matching",
                                     threshold=0.5, visualize_alone=False, g1digraph=False, g2digraph=False):
    fig, ax = plt.subplots(figsize=(12, 6))  # Single axis
    plt.title(image_name)

    # Get nodes and edges data
    if g1digraph:
        nodes_data1 = graph1.nodes(data=True)
        edges_data1 = graph1.edges(data=True)
    else:
        nodes_data1 = graph1.get_attributes_of_all_nodes()
        edges_data1 = graph1.get_attributes_of_all_edges()

    if g2digraph:
        nodes_data2 = graph2.nodes(data=True)
        edges_data2 = graph2.edges(data=True)
    else:
        nodes_data2 = graph2.get_attributes_of_all_nodes()
        edges_data2 = graph2.get_attributes_of_all_edges()

    # Offset for the second graph
    offset_value = 10  # Adjusted spacing for clarity
    offset = np.array([offset_value, 0])  # 2D offset

    # Store node positions
    pos1, pos2 = {}, {}

    # Plot first graph (graph1)
    for node_data in nodes_data1:
        node_id, node_attr = node_data[0], node_data[1]
        center = np.array(node_attr["center"])[:2]  # Ensure 2D coordinates
        pos1[node_id] = center  # Save for matching

        # Plot nodes
        if node_attr["viz"]["type"] == "Point":
            ax.plot(center[0], center[1], node_attr["viz"]["feat"], markersize=5)
        elif node_attr["viz"]["type"] == "Line":
            viz_data = np.array(node_attr["viz"]["limits"])
            ax.plot(viz_data[:, 0], viz_data[:, 1], node_attr["viz"]["feat"])

    # Plot second graph (graph2) with offset
    for node_data in nodes_data2:
        node_id, node_attr = node_data[0], node_data[1]
        center = np.array(node_attr["center"])[:2]  # Ensure 2D coordinates

        pos2[node_id] = center + offset  # Apply offset to shift the second graph

        # Plot nodes
        if node_attr["viz"]["type"] == "Point":
            ax.plot(center[0] + offset[0], center[1] + offset[1], node_attr["viz"]["feat"], markersize=5)
        elif node_attr["viz"]["type"] == "Line":
            viz_data = np.array(node_attr["viz"]["limits"])
            ax.plot(viz_data[:, 0] + offset[0], viz_data[:, 1] + offset[1], node_attr["viz"]["feat"])

    # Plot edges for both graphs
    for edge_data in edges_data1:
        if edge_data[0] in pos1 and edge_data[1] in pos1:
            points = np.array([pos1[edge_data[0]], pos1[edge_data[1]]])
            ax.plot(points[:, 0], points[:, 1], edge_data[2]["viz_feat"])

    for edge_data in edges_data2:
        if edge_data[0] in pos2 and edge_data[1] in pos2:
            points = np.array([pos2[edge_data[0]], pos2[edge_data[1]]])
            ax.plot(points[:, 0], points[:, 1], edge_data[2]["viz_feat"])

    # Draw matching connections between graphs
    for i, row in enumerate(match_result):
        for j, value in enumerate(row):
            if value > threshold:  # Only show strong matches
                if i in pos1 and j in pos2:
                    x_values = [pos1[i][0], pos2[j][0]]
                    y_values = [pos1[i][1], pos2[j][1]]
                    ax.plot(x_values, y_values, 'k--', alpha=0.5)  # Dashed matching line

    # Set aspect ratio
    ax.set_aspect('equal', adjustable='datalim')

    # Display or save
    plt.draw()
    if visualize_alone:
        plt.show()
    else:
        plt.close(fig)

    return fig

# Function to visualize graph matching results
def plot_matching(graph1, graph2, match_result, threshold=0.5):
    pos1 = nx.spring_layout(graph1)  # Layout for the first graph
    pos2 = nx.spring_layout(graph2)  # Layout for the second graph

    # Offset to separate the two graphs
    offset = 2
    for key in pos2:
        pos2[key] += np.array([offset, 0])  # Shift the second graph to the right

    plt.figure(figsize=(10, 6))

    # Draw the first graph
    nx.draw(graph1, pos1, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, font_size=10)
    # Draw the second graph
    nx.draw(graph2, pos2, with_labels=True, node_color="lightcoral", edge_color="gray", node_size=500, font_size=10)

    # Draw matching connections between nodes
    for i, row in enumerate(match_result):
        for j, value in enumerate(row):
            if value > threshold:  # Show only strong matches
                plt.plot([pos1[i][0], pos2[j][0]], [pos1[i][1], pos2[j][1]], 'k--', alpha=0.5)

    plt.title("Graph Matching")
    plt.show()

# Function to visualize graph matching results as in the PyGmTools example
def draw_graphs(graph1, graph2):
    """
    Draws two graphs side by side using NetworkX graph objects.

    Parameters:
        graph1 (networkx.Graph or networkx.DiGraph): First graph.
        graph2 (networkx.Graph or networkx.DiGraph): Second graph.
    """
    plt.figure(figsize=(8, 4))

    # Convert graphs to adjacency matrices
    A1 = nx.to_numpy_array(graph1)
    A2 = nx.to_numpy_array(graph2)

    # Convert adjacency matrices back to NetworkX graphs
    G1 = nx.from_numpy_array(A1, create_using=nx.DiGraph if isinstance(graph1, nx.DiGraph) else nx.Graph)
    G2 = nx.from_numpy_array(A2, create_using=nx.DiGraph if isinstance(graph2, nx.DiGraph) else nx.Graph)

    # Generate layouts
    pos1 = nx.spring_layout(G1)
    pos2 = nx.spring_layout(G2)

    # Draw first graph
    plt.subplot(1, 2, 1)
    plt.title('Graph 1')
    nx.draw_networkx(G1, pos=pos1)

    # Draw second graph
    plt.subplot(1, 2, 2)
    plt.title('Graph 2')
    nx.draw_networkx(G2, pos=pos2)

    plt.show()



def draw_graph_matching(graph1, graph2, match_result, X_gt=None, threshold=0.5):
    """
    Draws two graphs side by side and visualizes node correspondences using a matching matrix.

    Parameters:
        graph1 (networkx.Graph or networkx.DiGraph): First graph.
        graph2 (networkx.Graph or networkx.DiGraph): Second graph.
        match_result (np.array or torch.Tensor): Matching matrix.
        X_gt (np.array or torch.Tensor, optional): Ground truth matching matrix.
        threshold (float): Minimum value to consider a match.
    """
    plt.figure(figsize=(8, 4))

    # Convert graphs to adjacency matrices
    A1 = nx.to_numpy_array(graph1)
    A2 = nx.to_numpy_array(graph2)

    # Convert adjacency matrices back to NetworkX graphs
    G1 = nx.from_numpy_array(A1, create_using=nx.DiGraph if isinstance(graph1, nx.DiGraph) else nx.Graph)
    G2 = nx.from_numpy_array(A2, create_using=nx.DiGraph if isinstance(graph2, nx.DiGraph) else nx.Graph)

    # Generate layouts
    pos1 = nx.spring_layout(G1)
    pos2 = nx.spring_layout(G2)

    # Convert match_result and X_gt to NumPy if they are Torch tensors
    if isinstance(match_result, torch.Tensor):
        match_result = match_result.detach().cpu().numpy()
    if X_gt is not None and isinstance(X_gt, torch.Tensor):
        X_gt = X_gt.detach().cpu().numpy()

    # Draw graphs
    ax1 = plt.subplot(1, 2, 1)
    plt.title('Graph 1')
    nx.draw_networkx(G1, pos=pos1)

    ax2 = plt.subplot(1, 2, 2)
    plt.title('Graph 2')
    nx.draw_networkx(G2, pos=pos2)

    num_nodes = match_result.shape[0]

    # Draw matching connections
    for i in range(num_nodes):
        j = np.argmax(match_result[i])  # Get best match for node i
        if match_result[i, j] > threshold:  # Only consider strong matches
            match_color = "green" if X_gt is not None and X_gt[i, j] else "red"
            con = ConnectionPatch(xyA=pos1[i], xyB=pos2[j], coordsA="data", coordsB="data",
                                  axesA=ax1, axesB=ax2, color=match_color, alpha=0.6, linewidth=1)
            plt.gca().add_artist(con)

    plt.show()


def draw_aligned_graph(graph1, graph2, match_result, X_gt=None, threshold=0.5):
    """
    Draws Graph 1 and an aligned version of Graph 2 based on the matching matrix.

    Parameters:
        graph1 (networkx.Graph or networkx.DiGraph): First graph.
        graph2 (networkx.Graph or networkx.DiGraph): Second graph.
        match_result (np.array or torch.Tensor): Matching matrix.
        X_gt (np.array or torch.Tensor, optional): Ground truth matching matrix.
        threshold (float): Minimum value to consider a match.
    """
    plt.figure(figsize=(8, 4))

    # Convert graphs to adjacency matrices
    A1 = nx.to_numpy_array(graph1)
    A2 = nx.to_numpy_array(graph2)

    # Convert adjacency matrices back to NetworkX graphs
    G1 = nx.from_numpy_array(A1, create_using=nx.DiGraph if isinstance(graph1, nx.DiGraph) else nx.Graph)
    G2 = nx.from_numpy_array(A2, create_using=nx.DiGraph if isinstance(graph2, nx.DiGraph) else nx.Graph)

    # Generate layouts
    pos1 = nx.spring_layout(G1)  # Layout for Graph 1
    pos2 = nx.spring_layout(G2)  # Default layout for Graph 2 (for unmatched nodes)

    # Convert match_result and X_gt to NumPy if they are Torch tensors
    if isinstance(match_result, torch.Tensor):
        match_result = match_result.detach().cpu().numpy()
    if X_gt is not None and isinstance(X_gt, torch.Tensor):
        X_gt = X_gt.detach().cpu().numpy()

    # Create aligned positions
    align_pos2 = {}
    num_nodes = match_result.shape[0]

    for i in range(num_nodes):
        j = np.argmax(match_result[i])  # Best match for node i
        if match_result[i, j] > threshold:  # Only consider strong matches
            align_pos2[j] = pos1[i]  # Move node j to match node i

    # Ensure all nodes in G2 have a position (either aligned or default)
    for node in G2.nodes():
        if node not in align_pos2:
            align_pos2[node] = pos2[node]  # Assign default position

    ax1 = plt.subplot(1, 2, 1)
    plt.title('Graph 1')
    nx.draw_networkx(G1, pos=pos1)

    ax2 = plt.subplot(1, 2, 2)
    plt.title('Aligned Graph 2')
    nx.draw_networkx(G2, pos=align_pos2)

    # Draw matching connections
    for i in range(num_nodes):
        j = np.argmax(match_result[i])  # Get best match for node i
        if match_result[i, j] > threshold:  # Only consider strong matches
            match_color = "green" if X_gt is not None and X_gt[i, j] else "red"
            con = ConnectionPatch(xyA=pos1[i], xyB=align_pos2[j], coordsA="data", coordsB="data",
                                  axesA=ax1, axesB=ax2, color=match_color, alpha=0.6, linewidth=1)
            plt.gca().add_artist(con)

    plt.show()