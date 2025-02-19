import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch
from matplotlib.patches import ConnectionPatch



def visualize_nxgraph(graph, image_name, visualize_alone=False):
    nodes_data = graph.get_attributes_of_all_nodes()
    fig = plt.figure(image_name)
    # fig.clf()
    ax = fig.add_subplot(111)
    for node_data in nodes_data:
        if node_data[1]["viz_type"] == "Point":
            ax.plot(node_data[1]["viz_data"][0], node_data[1]["viz_data"][1], node_data[1]["viz_feat"])
            # print(f'dbg node_data[1] {node_data[1]}')

        elif node_data[1]["viz_type"] == "Line":
            viz_data = np.array(node_data[1]["viz_data"])
            linewidth = node_data[1]["linewidth"] if "linewidth" in node_data[1].keys() else 1.5
            
            ax.plot(viz_data[:,0], viz_data[:,1], node_data[1]["viz_feat"], linewidth=linewidth)

            norm_line = np.stack([node_data[1]["center"], node_data[1]["center"] + node_data[1]["normal"]/4])
            ax.plot(norm_line[:,0], norm_line[:,1], "b", linewidth=linewidth)

    edges_data = graph.get_attributes_of_all_edges()
    for edge_data in edges_data:
        points = np.array([nodes_data[edge_data[0]]["center"], nodes_data[edge_data[1]]["center"]])
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
        if node_data[1]["viz_type"] == "Point":
            ax1.plot(node_data[1]["viz_data"][0], node_data[1]["viz_data"][1], node_data[1]["viz_feat"])
        elif node_data[1]["viz_type"] == "Line":
            viz_data = np.array(node_data[1]["viz_data"])
            ax1.plot(viz_data[:,0], viz_data[:,1], node_data[1]["viz_feat"])
    for node_data in nodes_data2:
        if node_data[1]["viz_type"] == "Point":
            ax2.plot(node_data[1]["viz_data"][0], node_data[1]["viz_data"][1], node_data[1]["viz_feat"])
        elif node_data[1]["viz_type"] == "Line":
            viz_data = np.array(node_data[1]["viz_data"])
            ax2.plot(viz_data[:,0], viz_data[:,1], node_data[1]["viz_feat"])
    
    for edge_data in edges_data1:
        points = np.array([nodes_data1[edge_data[0]]["center"], nodes_data1[edge_data[1]]["center"]])
        ax1.plot(points[:,0], points[:,1], edge_data[2]["viz_feat"])
    for edge_data in edges_data2:
        points = np.array([nodes_data2[edge_data[0]]["center"], nodes_data2[edge_data[1]]["center"]])
        ax2.plot(points[:,0], points[:,1], edge_data[2]["viz_feat"])
    
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
        if node_attr["viz_type"] == "Point":
            ax.plot(center[0], center[1], node_attr["viz_feat"], markersize=5)
        elif node_attr["viz_type"] == "Line":
            viz_data = np.array(node_attr["viz_data"])
            ax.plot(viz_data[:, 0], viz_data[:, 1], node_attr["viz_feat"])

    # Plot second graph (graph2) with offset
    for node_data in nodes_data2:
        node_id, node_attr = node_data[0], node_data[1]
        center = np.array(node_attr["center"])[:2]  # Ensure 2D coordinates

        pos2[node_id] = center + offset  # Apply offset to shift the second graph

        # Plot nodes
        if node_attr["viz_type"] == "Point":
            ax.plot(center[0] + offset[0], center[1] + offset[1], node_attr["viz_feat"], markersize=5)
        elif node_attr["viz_type"] == "Line":
            viz_data = np.array(node_attr["viz_data"])
            ax.plot(viz_data[:, 0] + offset[0], viz_data[:, 1] + offset[1], node_attr["viz_feat"])

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