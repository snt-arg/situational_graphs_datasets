from SyntheticDatasetGenerator import SyntheticDatasetGenerator
from graph_visualizer import visualize_nxgraph, visualize_nxgraph_3d
import matplotlib.pyplot as plt
import json, os, time, sys
import networkx as nx


from graph_datasets.config import get_config as get_datasets_config
# from graph_reasoning.config import get_config as get_reasoning_config
synteticdataset_settings = get_datasets_config("msd")

# synteticdataset_settings["source"]["base_graphs"]["n_buildings"] = 10
dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings, logger = None, report_path = "???", dataset_name = "test")
# dataset_generator.create_dataset()
# filtered_nxdataset = dataset_generator.get_filtered_datset(settings_hdata["nodes"],settings_hdata["edges"])["noise"]
extended_nxdatset = dataset_generator.extend_nxdataset(dataset_generator.graphs["original"], "training", "training")

all_dataset = extended_nxdatset["test"] + extended_nxdatset["test"] +extended_nxdatset["val"]


for graph in all_dataset[:10]:
    visualize_nxgraph_3d(graph, "train data", visualize_alone=True, include_node_ids=False)
    plt.show()

    # centers_2d = centers = nx.get_node_attributes(graph.graph, 'center')
    # labels = nx.get_node_attributes(graph.graph, 'type')
    # pos = nx.spring_layout(graph.graph)
    # edge_weights = nx.get_edge_attributes(graph.graph, 'weight')
    # nx.draw(graph.graph, with_labels=True, labels=labels, node_color='lightblue', node_size=800, font_size=14)
    # nx.draw_networkx_edge_labels(graph.graph, centers_2d, edge_labels=edge_weights)
    # plt.title("Graph Visualization")
    # plt.show()


#     \acp{3DSG} organize environments into entities and relations across hierarchical layers, providing a compact, queryable substrate for perception, mapping, and planning \cite{bae2022survey}. Early systems were largely offline and object-centric, building room/building abstractions from RGB detections \cite{3d_scene_graph} or inferring object–object relations from semantic instance cues in real time \cite{sgf}. Open-vocabulary pipelines recently leverage vision–language models to populate graphs with long-tail categories \cite{gu2023conceptgraphs,koch2024open3dsg}; however, these graphs are typically flat (non-hierarchical) and do not induce emergent structural nodes such as \textit{rooms} or \textit{walls}. In robotics, hierarchical, online 3DSGs have been pushed by Hydra, which constructs a four-layer scene graph and identifies rooms via free-space voxel clustering \cite{hydra}, later extended to categorize building areas \cite{hughes2023foundations,talak2021neural}. Outdoor 3DSGs similarly extract roads/intersections with rule-based heuristics \cite{greve2023collaborative}. The LiDAR-based \textit{S-Graphs+} integrates a hierarchical 3DSG into a SLAM back end and demonstrates that semantic–geometric coupling improves odometry and mapping \cite{s_graphs+}, with extensions to visual input and multi-floor optimization \cite{vsgraphs,multifloorsgraphs}. Parallel but orthogonal lines segment rooms directly from 2D/3D maps without explicit scene graphs; while effective for layout parsing, they do not expose the relational hierarchy needed for reasoning and factor-graph coupling. However, most scene-graph pipelines still instantiate emergent nodes using ad-hoc clustering or pairwise heuristics with strong layout assumptions, and their continuous spatial attributes (e.g., centroids, normals, extents) are typically hand-specified per concept/edge type, limiting scalability, temporal consistency, and sensor-agnostic operation.

# Beyond robotics, graph generation has examined how to model attributes jointly with topology. Adversarial approaches (e.g., NANG) synthesize node attributes that align with graph structure \cite{chen2019node}; disentangled formulations seek to factor topology from attributes for controllability; and controlled generation specifies dependencies between classes, attributes, and edges. Complementarily, \acp{GNN} can regress continuous node attributes and invariant relational quantities directly on graphs, enabling differentiable, end-to-end pipelines \cite{fang2022invariant}. In robotic 3DSGs, learned pairwise classification between primitives (e.g., \textit{plane–plane}) has been used to infer \textit{walls}/\textit{rooms} followed by community clustering, often under restricted layouts (e.g., rectangular rooms) \cite{reasoning_v1}. From a metric perspective, every node in a situational/metric 3DSG possesses a pose or centroid; thus, continuous spatial attributes are not auxiliary but a first-class output of the generation step, and should be estimated jointly with semantics and relations and maintained over time as the SLAM state evolves. Yet existing methods rarely enforce real-time metric–topology coherence at scale, seldom treat continuous spatial attributes as integral to scene-graph generation, and offer limited mechanisms for uncertainty-aware, temporally consistent updates required for online mapping.

# Factor graphs are the standard abstraction for large-scale probabilistic inference in SLAM, enabling heterogeneous constraints to be fused via sparse nonlinear least squares \cite{dellaert2017factor}. Recent efforts couple scene-graph structure to estimation—for example, constrained factor graphs that inject semantic relations or pipelines that import scene-graph cues into the optimizer \cite{haroon2024constrained,taylor2024factor}. \textit{S-Graphs+} integrates a hierarchical 3DSG directly into the optimization state and shows clear gains, but relies on manually specified factor forms and covariances for each concept (e.g., room–plane, wall–plane), which hinders extensibility when new emergent nodes or relations are introduced \cite{s_graphs+}. In parallel, the learning community has explored GNNs that predict continuous, invariant factors or potentials from graph structure \cite{fang2022invariant}. From the uncertainty side, Bayesian approximations such as Monte-Carlo dropout provide epistemic variance estimates for neural predictions, and calibration (e.g., temperature scaling) improves the reliability of categorical confidences \cite{ryu2019bayesian}. Some SLAM systems down-weight constraints using external semantic confidence or robustification strategies, but they stop short of learning geometric factor functions and their noise models from data, and they do not tie those uncertainties to the scene-graph generation process itself (e.g., relation confidence + geometric regression uncertainty) in a principled way. However, no prior work jointly learns geometric factor functions \emph{and} their covariances for emergent 3DSG concepts, yielding uncertainty-aware constraints that are directly optimizable within SLAM.