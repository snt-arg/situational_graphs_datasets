import matplotlib.pyplot as plt
import numpy as np


def visualize_nxgraph(graph, image_name, visualize_alone=False, include_node_ids=True, logger = None):
    nodes_data = graph.get_attributes_of_all_nodes()
    fig = plt.figure(image_name)
    # fig.clf()
    ax = fig.add_subplot(111)
    for node_data in nodes_data:
        if node_data[1]["viz_type"] == "Point":
            if "markersize" in node_data[1].keys():
                markersize = node_data[1]["markersize"]
            else:
                markersize = 1.0
            ax.plot(node_data[1]["viz_data"][0], node_data[1]["viz_data"][1], node_data[1]["viz_feat"], markersize=markersize*10)
            tag_center = node_data[1]["viz_data"]

        elif node_data[1]["viz_type"] == "Line":
            viz_data = np.array(node_data[1]["viz_data"])
            linewidth = node_data[1]["linewidth"] if "linewidth" in node_data[1].keys() else 1.5
            
            ax.plot(viz_data[:,0], viz_data[:,1], node_data[1]["viz_feat"], linewidth=linewidth)

            norm_line = np.stack([node_data[1]["center"], node_data[1]["center"] + node_data[1]["normal"]/4])
            ax.plot(norm_line[:,0], norm_line[:,1], "b", linewidth=linewidth)
            tag_center = np.array(node_data[1]["center"]) + np.array(node_data[1]["normal"]) * 0.5

        if include_node_ids:
            plt.text(tag_center[0], tag_center[1], str(node_data[0]), fontsize=12, color='black')

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

def visualize_nxgraph_3d(graph, image_name, visualize_alone=False, include_node_ids=True, logger=None):
    nodes_data = graph.get_attributes_of_all_nodes()
    fig = plt.figure(image_name)
    ax = fig.add_subplot(111, projection='3d')

    def to_3d(arr):
        arr = np.array(arr)
        if arr.shape[-1] == 2:
            arr = np.append(arr, 0)
        return arr

    # For legend
    legend_handles = {}
    for node_data in nodes_data:
        if node_data[1]["viz_type"] == "Point":
            markersize = node_data[1].get("markersize", 1.0)
            viz_data = to_3d(node_data[1]["viz_data"])
            color = _mpl_color_from_feat(node_data[1]["viz_feat"])
            marker = node_data[1]["viz_feat"][1] if len(node_data[1]["viz_feat"]) > 1 else 'o'
            label = node_data[1].get("type", "Point")
            # Only add one handle per label
            if label not in legend_handles:
                h = ax.scatter([], [], [], marker=marker, s=markersize*30, color=color, label=label)
                legend_handles[label] = h
            ax.scatter(viz_data[0], viz_data[1], viz_data[2], marker=marker, s=markersize*30, color=color)
            tag_center = viz_data
        elif node_data[1]["viz_type"] == "Line":
            viz_data = np.array(node_data[1]["viz_data"])
            linewidth = node_data[1].get("linewidth", 1.5)
            color = _mpl_color_from_feat(node_data[1]["viz_feat"])
            label = node_data[1].get("type", "Line")
            if viz_data.shape[1] == 2:
                viz_data = np.hstack([viz_data, np.zeros((viz_data.shape[0], 1))])
            # Only add one handle per label
            if label not in legend_handles:
                h, = ax.plot([], [], [], color=color, linewidth=linewidth, label=label)
                legend_handles[label] = h
            ax.plot(viz_data[:,0], viz_data[:,1], viz_data[:,2], color=color, linewidth=linewidth)
            center = to_3d(node_data[1]["center"])
            normal = to_3d(node_data[1].get("normal", [0, 0, 0]))
            norm_line = np.stack([center, center + normal/4])
            ax.plot(norm_line[:,0], norm_line[:,1], norm_line[:,2], color='b', linewidth=linewidth)
            tag_center = center + normal * 0.5 if np.linalg.norm(normal) > 0 else center
        if include_node_ids:
            ax.text(tag_center[0], tag_center[1], tag_center[2], str(node_data[0]), fontsize=10, color='black')
    edges_data = graph.get_attributes_of_all_edges()
    for edge_data in edges_data:
        points = np.array([to_3d(nodes_data[edge_data[0]]["center"]), to_3d(nodes_data[edge_data[1]]["center"])])
        color = _mpl_color_from_feat(edge_data[2].get("viz_feat", "k"))
        linewidth = edge_data[2].get("linewidth", 1.5)
        alpha = edge_data[2].get("alpha", 1.0)
        label = edge_data[2].get("type", "Edge")
        # Only add one handle per label
        if label not in legend_handles:
            h, = ax.plot([], [], [], color=color, linewidth=linewidth, alpha=alpha, label=label)
            legend_handles[label] = h
        ax.plot(points[:,0], points[:,1], points[:,2], color=color, linewidth=linewidth, alpha=alpha)
        if "pred" in edge_data[2]:
            center = (points[0] + points[1]) / 2
            ax.text(center[0], center[1], center[2], "{:.2f}".format(edge_data[2]['pred']), fontsize=9)
    ax.set_box_aspect([1,1,1])
    # Add legend
    ax.legend()
    if visualize_alone:
        plt.show()
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