import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


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

def visualize_digraph(graph, image_name):
    plt.figure(figsize=(8, 4))
    plt.title(image_name)
    nx.draw_networkx(graph)
    plt.show()