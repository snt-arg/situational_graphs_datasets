import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np

def create_graph_figure(graph):
    # Retrieve node and edge data from your graph object.
    # nodes_data: list of (node_id, attr)
    # edges_data: list of (node_id1, node_id2, attr)
    nodes_data = graph.get_attributes_of_all_nodes()
    edges_data = graph.get_attributes_of_all_edges()
    
    # We'll build separate traces for different visual elements.
    
    # --- For Node Interactivity ---
    # For "Point" nodes:
    point_node_x, point_node_y, point_node_text = [], [], []
    # For "Line" nodes, we add a clickable marker at the computed proxy position.
    line_node_x, line_node_y, line_node_text = [], [], []
    
    # --- For Node Drawing (non-interactive) ---
    # For "Line" nodes, we draw the main line and the normal line.
    line_draw_traces = []  # Each element will be a Scatter trace.
    normal_line_traces = []
    
    # Process each node
    for node_id, attr in nodes_data:
        if attr["viz_type"] == "Point":
            # Use the provided "viz_data" as the coordinate (take only first 2 dims)
            coords = np.array(attr["viz_data"])[:2]
            point_node_x.append(coords[0])
            point_node_y.append(coords[1])
            text = f"Node {node_id}<br>Type: Point<br>Coord: {attr['viz_data'][:2]}"
            point_node_text.append(text)
        elif attr["viz_type"] == "Line":
            # For drawing, we use the full line data
            viz_data = np.array(attr["viz_data"])[:, :2]  # assume it's 2D data
            linewidth = attr.get("linewidth", 1.5)
            # Create a trace for the main line (use the provided viz_feat for style)
            line_trace = go.Scatter(
                x=viz_data[:,0],
                y=viz_data[:,1],
                mode='lines',
                line=dict(width=linewidth, color='green'),  # you might want to parse attr["viz_feat"] further
                hoverinfo='skip'  # non-interactive
            )
            line_draw_traces.append(line_trace)
            
            # Draw the normal line (from center to center+normal/4) in blue
            center = np.array(attr["center"])[:2]
            normal = np.array(attr["normal"])[:2]
            norm_line = np.vstack([center, center + normal/4])
            normal_trace = go.Scatter(
                x=norm_line[:,0],
                y=norm_line[:,1],
                mode='lines',
                line=dict(width=linewidth, color='blue'),
                hoverinfo='skip'
            )
            normal_line_traces.append(normal_trace)
            
            # For interactivity, use a proxy marker at center+normal*0.5
            tag_center = (np.array(attr["center"]) + np.array(attr["normal"])*0.5)[:2]
            line_node_x.append(tag_center[0])
            line_node_y.append(tag_center[1])
            text = f"Node {node_id}<br>Type: Line<br>Center: {attr['center'][:2]}"
            line_node_text.append(text)
    
    # Create scatter traces for interactive nodes:
    point_nodes_trace = go.Scatter(
        x=point_node_x,
        y=point_node_y,
        mode='markers+text',
        marker=dict(size=12, color='red', line=dict(width=1, color='black')),
        text=[str(i) for i in [node_id for node_id, attr in nodes_data if attr["viz_type"]=="Point"]],
        hoverinfo='text',
        hovertext=point_node_text,
        name='Point Nodes'
    )
    line_nodes_trace = go.Scatter(
        x=line_node_x,
        y=line_node_y,
        mode='markers+text',
        marker=dict(size=12, color='orange', line=dict(width=1, color='black')),
        text=[str(i) for i in [node_id for node_id, attr in nodes_data if attr["viz_type"]=="Line"]],
        hoverinfo='text',
        hovertext=line_node_text,
        name='Line Nodes'
    )
    
    # --- For Edges ---
    # We draw edges using the centers of the nodes.
    edge_x, edge_y, edge_text = [], [], []
    for edge in edges_data:
        node_id1, node_id2, attr = edge
        # Use get_attributes_of_node to retrieve each node's attributes.
        node1 = graph.get_attributes_of_node(node_id1)
        node2 = graph.get_attributes_of_node(node_id2)
        p1 = np.array(node1["center"])[:2]
        p2 = np.array(node2["center"])[:2]
        edge_x.extend([p1[0], p2[0], None])
        edge_y.extend([p1[1], p2[1], None])
        text = f"Edge ({node_id1}-{node_id2})"
        if "pred" in attr:
            text += f"<br>Pred: {attr['pred']}"
        edge_text.append(text)
    # Note: With a single trace for all edges, detailed per-edge hover text is limited.
    # For simplicity, we use a generic hover text.
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=2, color='gray'),
        hoverinfo='text',
        hovertext="Edge",  # generic message; you can extend this if needed
        name='Edges'
    )
    
    # --- Assemble all traces into a figure ---
    data = []
    data.append(edge_trace)
    # Add non-interactive traces first (lines for "Line" nodes)
    data.extend(line_draw_traces)
    data.extend(normal_line_traces)
    # Add interactive node markers on top
    data.append(point_nodes_trace)
    data.append(line_nodes_trace)
    
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title="Interactive Network Graph",
            showlegend=True,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            clickmode='event+select'
        )
    )
    
    return fig

def run_dash_app(graph):
    """Launch the Dash app using the provided graph."""
    app = dash.Dash(__name__)
    fig = create_graph_figure(graph)
    
    app.layout = html.Div([
        dcc.Graph(
            id='graph',
            figure=fig,
            style={'height': '80vh'}
        ),
        html.Div(id='output', style={'whiteSpace': 'pre-line', 'padding': '20px'})
    ])
    
    @app.callback(
        Output('output', 'children'),
        Input('graph', 'clickData')
    )
    def display_click_info(clickData):
        if clickData is None:
            return "Click on a node or edge to see its details."
        point = clickData['points'][0]
        curve = point.get('curveNumber', None)
        # Based on our trace order:
        # curveNumber 0: edges, 
        # then 1...: non-interactive line drawings (we ignore these),
        # last two traces are interactive nodes: assume point nodes and line nodes.
        if curve in [len(fig.data)-2, len(fig.data)-1]:
            info = point.get('hovertext', '')
            print(f"Clicked on node: {info}")
            return f"Clicked on node:\n{info}"
        elif curve == 0:
            info = "Edge clicked. (Detailed info not available in this view.)"
            print(info)
            return info
        else:
            return "Clicked on an unrecognized element."
    
    app.run_server(debug=True)

# ---------------------------
# To use:
# Import this module in your code, provide your graph object, and then call run_dash_app.
# For example:
#
# from interactive_graph_app import run_dash_app
# from your_graph_module import your_graph  # your graph object implementing the required methods
# run_dash_app(your_graph)
