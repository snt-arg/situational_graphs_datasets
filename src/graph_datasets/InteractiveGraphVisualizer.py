import matplotlib.pyplot as plt
import numpy as np
import queue

class InteractiveGraphVisualizer:
    def __init__(self, graph, image_name, group_queue, graph_update_queue, 
                 callback=None, include_node_ids=True, logger=None):
        """
        Parameters:
         - graph: your graph object implementing required methods. May be None initially.
         - image_name: title for the figure.
         - group_queue: a thread-safe queue to send finalized groups.
         - graph_update_queue: a thread-safe queue from which to retrieve new graph objects.
         - callback: an optional function to call when a group is finalized.
         - include_node_ids: if True, node IDs are drawn.
         - logger: logger object with an info() method.
        """
        # Enable interactive mode.
        plt.ion()
        
        self.graph = graph
        self.image_name = image_name
        self.group_queue = group_queue
        self.graph_update_queue = graph_update_queue
        self.callback = callback
        self.include_node_ids = include_node_ids
        self.logger = logger

        self.current_group_type = "R"
        self.group_colors = {"R": "red", "r": "orange", "W": "brown", "w": "black"}
        self.active_groups = {}       # group type -> set of selected node IDs (active)
        self.finalized_groups = {}    # group type -> list of finalized groups (each a set)
        self.node_coords = {}         # node id -> 2D coordinate
        self.selection_artists = {}   # persistent scatter overlays
        self.pending_graph = None     # not used now, since we use the graph_update_queue

        self.fig = plt.figure(self.image_name)
        self.ax = self.fig.add_subplot(111)
        if self.graph:
            self.draw_graph()

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.title(f"Interactive Graph - Current group type: {self.current_group_type}")

        # Set up a timer that polls both queues.
        self.start_timer()

    def start_timer(self):
        """Starts a timer that polls the graph_update_queue and processes GUI events."""
        def timer_callback():
            # Check if a new graph is available, and update if no active selection exists.
            try:
                new_graph = self.graph_update_queue.get_nowait()
                # Only update if there is no active selection.
                selection_exists = any(self.active_groups.get(gt) for gt in self.active_groups)
                if not selection_exists:
                    if self.logger:
                        self.logger.info("Graph update applied from queue.")
                    self.graph = new_graph
                    self.draw_graph()
                else:
                    if self.logger:
                        self.logger.info("Graph update deferred due to active selection.")
                    # Optionally, you could re-put it into the queue if desired.
                    self.graph_update_queue.put(new_graph)
            except queue.Empty:
                pass
        # Create a matplotlib timer that fires every 100 ms.
        timer = self.fig.canvas.new_timer(interval=100)
        timer.add_callback(timer_callback)
        timer.start()

    def draw_graph(self):
        """Redraws the graph from self.graph onto self.ax and updates node coordinates."""
        self.ax.cla()
        self.node_coords = {}

        nodes_data = self.graph.get_attributes_of_all_nodes()
        for node_data in nodes_data:
            node_id, attr = node_data[0], node_data[1]
            if attr["viz_type"] == "Point":
                coords = np.array(attr["viz_data"])[:2]
            elif attr["viz_type"] == "Line":
                coords = np.array(attr["center"])[:2]
                viz_data = np.array(attr["viz_data"])[:, :2]
                linewidth = attr.get("linewidth", 1.5)
                self.ax.plot(viz_data[:,0], viz_data[:,1], attr["viz_feat"], linewidth=linewidth)
                center = np.array(attr["center"])[:2]
                normal = np.array(attr["normal"])[:2]
                norm_line = np.stack([center, center + normal/4])
                self.ax.plot(norm_line[:,0], norm_line[:,1], "b", linewidth=linewidth)
            else:
                continue
            self.ax.plot(coords[0], coords[1], attr["viz_feat"])
            self.node_coords[node_id] = coords
            if self.include_node_ids:
                self.ax.text(coords[0], coords[1], str(node_id), fontsize=12, color='black')

        edges_data = self.graph.get_attributes_of_all_edges()
        for edge_data in edges_data:
            node_id1, node_id2, attr = edge_data
            p1 = np.array(self.graph.get_attributes_of_node(node_id1)["center"])[:2]
            p2 = np.array(self.graph.get_attributes_of_node(node_id2)["center"])[:2]
            points = np.vstack([p1, p2])
            viz_feat = attr.get("viz_feat", "")
            linewidth = attr.get("linewidth", 1.5)
            alpha = attr.get("alpha", 1.0)
            self.ax.plot(points[:,0], points[:,1], viz_feat, linewidth=linewidth, alpha=alpha)
            if "pred" in attr:
                center_x = (points[0,0] + points[1,0]) / 2
                center_y = (points[0,1] + points[1,1]) / 2
                self.ax.text(center_x, center_y, "{:.2f}".format(attr['pred']))
        self.ax.set_aspect('equal', adjustable='datalim')
        self.update_selection()
        self.fig.canvas.draw_idle()

    def update_selection(self):
        """Updates or creates persistent scatter overlays for each active group."""
        for grp_type in list(self.selection_artists.keys()):
            if grp_type not in self.active_groups or not self.active_groups[grp_type]:
                self.selection_artists[grp_type].remove()
                del self.selection_artists[grp_type]
            else:
                offsets = np.array([self.node_coords[nid] for nid in self.active_groups[grp_type] if nid in self.node_coords])
                if offsets.size:
                    self.selection_artists[grp_type].set_offsets(offsets)
        for grp_type, group in self.active_groups.items():
            if group and grp_type not in self.selection_artists:
                offsets = np.array([self.node_coords[nid] for nid in group if nid in self.node_coords])
                if offsets.size:
                    scatter = self.ax.scatter(
                        offsets[:,0], offsets[:,1],
                        s=150, facecolors='none',
                        edgecolors=self.group_colors.get(grp_type, "red"),
                        linewidths=2, zorder=10, picker=False
                    )
                    self.selection_artists[grp_type] = scatter

    def on_click(self, event):
        """Uses display coordinates for hit detection and toggles node selection."""
        if event.inaxes != self.ax:
            return
        self.fig.canvas.draw()  # ensure transforms are updated
        click_disp = np.array([event.x, event.y])
        tolerance_pixels = 50.0
        for node_id, coords in self.node_coords.items():
            node_disp = self.ax.transData.transform(coords)
            distance = np.linalg.norm(click_disp - node_disp)
            if self.logger:
                self.logger.info(f"Click at {click_disp}, node {node_id} at {node_disp}, distance {distance:.2f} px")
            if distance < tolerance_pixels:
                if self.current_group_type not in self.active_groups:
                    self.active_groups[self.current_group_type] = set()
                if node_id in self.active_groups[self.current_group_type]:
                    self.active_groups[self.current_group_type].remove(node_id)
                    if self.logger:
                        self.logger.info(f"Node {node_id} removed from group '{self.current_group_type}'.")
                else:
                    self.active_groups[self.current_group_type].add(node_id)
                    if self.logger:
                        self.logger.info(f"Node {node_id} added to group '{self.current_group_type}'.")
                self.update_selection()
                self.fig.canvas.draw_idle()
                return
        if self.logger:
            self.logger.info("Background click detected; no group change (use Enter to finalize a group).")

    def on_key(self, event):
        """Handles key events to change group type or finalize the active group."""
        if event.key in self.group_colors:
            self.current_group_type = event.key
            if self.logger:
                self.logger.info(f"Current group type set to '{self.current_group_type}'.")
        elif event.key == "enter":
            if (self.current_group_type in self.active_groups and 
                self.active_groups[self.current_group_type]):
                if self.current_group_type not in self.finalized_groups:
                    self.finalized_groups[self.current_group_type] = []
                finalized = self.active_groups[self.current_group_type].copy()
                self.finalized_groups[self.current_group_type].append(finalized)
                if self.logger:
                    self.logger.info(f"Finalized group '{self.current_group_type}': {finalized}")
                # Put the finalized group into the group_queue so the node can process it.
                self.group_queue.put((self.current_group_type, finalized))
                if self.callback:
                    self.callback(self.current_group_type, finalized)
                self.active_groups[self.current_group_type] = set()
                self.update_selection()
                self.fig.canvas.draw_idle()
                # Also check for pending graph updates.
                try:
                    new_graph = self.graph_update_queue.get_nowait()
                    if self.logger:
                        self.logger.info("Applying pending graph update after finalizing selection.")
                    self.graph = new_graph
                    self.draw_graph()
                except queue.Empty:
                    pass
            else:
                if self.logger:
                    self.logger.info("No nodes selected in current group to finalize.")

    def update_graph(self, new_graph):
        """
        Updates the graph if no active selection exists; otherwise, defers update by placing
        the new graph into the graph_update_queue.
        """
        selection_exists = any(self.active_groups.get(gt) for gt in self.active_groups)
        if selection_exists:
            self.graph_update_queue.put(new_graph)
            if self.logger:
                self.logger.info("Graph update deferred due to active selection.")
        else:
            if self.logger:
                self.logger.info("Graph update applied immediately.")
            self.graph = new_graph
            self.draw_graph()
            self.fig.canvas.draw_idle()

    def show(self):
        """Displays the interactive graph window (blocking call)."""
        plt.show()

    def get_finalized_groups(self):
        return self.finalized_groups
