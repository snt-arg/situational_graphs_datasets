import matplotlib.pyplot as plt
import numpy as np
import pickle
import queue
import time
import copy
import os

from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from situational_graphs_datasets.graph_visualizer import _mpl_color_from_feat
from situational_graphs_wrapper.GraphWrapper import GraphWrapper

class InteractiveGraphVisualizer:
    def __init__(
            self,
            graph,
            image_name,
            group_queue,
            graph_update_queue,
            callback=None,
            logger=None,
            full_graph=None,
            default_save_dir=None,
            fig=None,
        ):
        """
        Parameters:
         - graph: your graph object implementing required methods. May be None initially.
         - image_name: title for the figure.
         - group_queue: a thread-safe queue to send finalized groups.
         - graph_update_queue: a thread-safe queue from which to retrieve new graph objects.
         - callback: an optional function to call when a group is finalized.
         - logger: logger object with an info() method.
        """
        # Enable interactive mode.
        # plt.ion()
        
        self.graph = graph
        self.image_name = image_name
        self.group_queue = group_queue
        self.graph_update_queue = graph_update_queue
        self.callback = callback
        self.include_node_ids = False
        self.logger = logger

        self.current_group_type = "R"
        self.group_colors = {"R": "red", "r": "orange", "W": "brown", "w": "black"}
        self.active_groups = {}       # group type -> set of selected node IDs (active)
        self.finalized_groups = {}    # group type -> list of finalized groups (each a set)
        self.node_coords = {}         # node id -> 2D coordinate
        self.selection_artists = {}   # persistent scatter overlays

        self.show_controls = False      # off initially due to it taking up a lot of screen real estate when the window is not fullscreen 
        self.show_edges = True
        self.show_normals = True
        self.show_node_centers = False  # useful debug tool
        self.show_working_edges = True
        self.full_graph = full_graph if full_graph is not None else graph  # Graph we want to persist in order to save

        self.default_save_dir = default_save_dir
        self.current_save_path = None

        # Optimization step
        self.node_ids_list = []         # index to Id mapping
        self.node_coords_array = None   # numpy array (N, 3)
        self.edge_indices = []          # List of tuples
        self.edge_keys = []             # List of actual keys

        # added to ensure logic from SDG
        # z offsets from SDG
        self.viz_center_offsets = {
            "ws": np.array([0, 0, 0]), 
            "room": np.array([0, 0, 2]), 
            "wall": np.array([0, 0, 1]),
            "floor": np.array([0, 0, 3]), 
            "building": np.array([0, 0, -2]), 
            "object": np.array([0, 0, 0.5]),
            "city": np.array([0, 0, -5])
        }

        # colors from SDG generation logic
        self.sdg_colors = {
            "ws": "red",      # ws nodes are usually handled specifically
            "room": "ro",
            "floor": "go",
            "building": "co",
            "wall": "mo",
            "city": "ko"      # City usually black
        }

        self.hovered_edge = None  # stores tuple (u,v) on hover
        self.hover_artist = None  # visual overlay 

        # gui related
        self.has_qt = False
        self.QtInputDialog = None

        self.current_z = 0.0         # current looking height 
        self.z_threshold = 1.5       # distance to show nodes (half of room height)
        self.use_z_filtering = False # start off as false to display whole graph

        self.pending_graph = None     # not used now, since we use the graph_update_queue

        # When `fig` is provided, embed into the caller's Figure (e.g. inside a
        # PyQt5 dashboard panel). Otherwise create a standalone figure via plt.
        self._embedded = fig is not None
        self.fig = fig if fig is not None else plt.figure(self.image_name)

        try:
            from PyQt5 import QtCore
            from PyQt5.QtWidgets import QInputDialog

            self.has_qt = True
            self.QtInputDialog = QInputDialog

            if not self._embedded and self.fig.canvas.manager.toolbar:
                win = self.fig.canvas.manager.window        # get window ojbects
                toolbar = self.fig.canvas.manager.toolbar   # get toolbar ojects

                # puts toolbar at the bottom like the standard TkAgg GUI manager
                win.addToolBar(QtCore.Qt.BottomToolBarArea, toolbar)
        except Exception:
            ########################### JUST PASS IF NOT USING QT #############################
            # Qt is actually highly recommended to replace the old TkAgg (Tkinter)            #
            # backend of matplotlib, specifically for 3D graphs                               #
            # Qt uses a highly optimized event loop, which handles rapid mouse movements      #
            # this makes mouse movements feel significantly less sluggish                     #
            # however Qt does not impact computation, it's just GUI related, so not necessary #
            ###################################################################################
            self.has_qt = False
            pass


        self.ax = self.fig.add_subplot(111, projection="3d")
        
        self._has_view = False  # used to preserve view 

        if self.graph:
            self.draw_graph(preserve_view=False)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)  # used for hightlights
        self.ax.set_title(f"Interactive Graph - Current group type: {self.current_group_type}")

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
                    # self.full_graph = new_graph  # keep the save file object in sync with the back end
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

    def _safe_recalc(self, g):
        """Call `recalculate_hierarchy_centers()` if available on this wrapper.

        The method only exists on the `feat/pard` branch of
        situational_graphs_wrapper; on other branches we silently skip it so
        delete/save still work.
        """
        if g is None:
            return
        fn = getattr(g, "recalculate_hierarchy_centers", None)
        if callable(fn):
            fn()

    def _ensure_3d(self, arr):
        """Helper to enusre coords are in 3D and flattened to output the correct shape"""
        if arr is None:
            return np.array([0.0, 0.0, 0.0])
        
        # convert to numpy array and flatten to get correct shape
        arr = np.array(arr, dtype=float).flatten()
        
        # if empty after conversion
        if len(arr) == 0:
            return np.array([0.0, 0.0, 0.0])

        # handle 2d cases (x, y), pad with 0 to make it (x, y, 0)
        if len(arr) < 3:
            arr = np.pad(arr, (0, 3 - len(arr)), mode='constant')
            
        # return exactly first 3 elements (x, y, z)
        return arr[:3]
    
    def _clear_highlights(self):
        """Removes all temp hover artist"""
        for artist in self.highlight_artists:
            try:
                artist.remove()
            except:
                pass
        self.highlight_artists = []

    def _draw_highlight_line(self, p1, p2, color, linewidth=2.0, style="-"):
        """Draws a single 3D line and adds it to the tracking list"""
        xs = [p1[0], p2[0]]
        ys = [p1[1], p2[1]]
        zs = [p1[2], p2[2]]
        lines = self.ax.plot(xs, ys, zs, c=color, linewidth=linewidth, linestyle=style, zorder=200)
        self.highlight_artists.extend(lines)

    def _project_coords(self):
        """Vectorized projection of all visible node coordinates to 2D screen space."""
        if self.node_coords_array is None or len(self.node_coords_array) == 0:
            return np.empty((0, 2))

        # get Projection Matrix
        M = self.ax.get_proj()
        
        # vectorized projection (Numpy is C-speed)
        xs, ys, zs = proj3d.proj_transform(
            self.node_coords_array[:, 0], 
            self.node_coords_array[:, 1], 
            self.node_coords_array[:, 2], 
            M
        )
        
        # convert to screen pixels using ax.transData
        screen_input = np.column_stack([xs, ys])
        screen_coords = self.ax.transData.transform(screen_input)
        
        return screen_coords
    
    def draw_graph(self, preserve_view=False):
        """
        Redraws the graph in 3D using optimized batching.

        Z-filtering logic:
        - if off: show everything
        - if on:
          - show node if it is currently selected, regardless of Z
          - show node if it has a specific hierarchy (currently building and city are always shown)
          - show nodes if they are within the current Z-slice range (essentially only that one level)
            - note: dependency check is performed for floor node to prevent visual bleeding of floors across different Z levels
          - otherwise hide everything else not on the current z level
        """
        
        # capture full 3d camera state (to preserve view after redraw)
        saved_cam_state = None
        if preserve_view and self._has_view:
            saved_cam_state = {
                "elev": self.ax.elev,
                "azim": self.ax.azim,
                "xlim": self.ax.get_xlim3d(),
                "ylim": self.ax.get_ylim3d(),
                "zlim": self.ax.get_zlim3d()
            }

        self.ax.cla()  # clears entire graph on each draw and uses a lot of performance

        # reset arrays
        self.node_ids_list = []
        self.node_coords_array = None
        self.edge_indices = []
        self.edge_keys = []
        self.node_coords = {}
        self.selection_artists = {}
        self.highlight_artists = [] 
        self.edge_cache = []    # not strictly needed with new logic, but kept for safety

        # batches
        scatter_batches = {}  # key: (marker, size) -> val: {"points": [], "colors": []}
        segments = []         # for Line3DCollection
        edge_colors = []
        
        # helpers
        node_id_to_idx = {}
        temp_coords_list = []
        legend_handles = {}
        visible_node_ids = set()

        nodes_data = self.graph.get_attributes_of_all_nodes()
        edges_data = self.graph.get_attributes_of_all_edges()

        # visibility logic 
        all_selected_ids = set()
        for grp in self.active_groups.values():
            all_selected_ids.update(grp)

        # define global hierarchy 
        global_hierarchy_types = {"building", "city"}

        # temp sets for dependency check
        floor_candidates = set()
        visible_room_ids = set()

        for node_id, attr in nodes_data:
            node_type = attr.get("type", "unknown")
            center = self._ensure_3d(attr.get("center", [0,0,0]))

            # always show selected node
            if node_id in all_selected_ids:
                visible_node_ids.add(node_id)
                if node_type == "room": visible_room_ids.add(node_id)
                continue

            # always show global hierarchy
            if node_type in global_hierarchy_types:
                visible_node_ids.add(node_id)
                continue

            # floor
            if node_type == "floor":
                floor_candidates.add(node_id)
                if not self.use_z_filtering: visible_node_ids.add(node_id)
                continue

            # standard nodes (ws, wall, room)
            if not self.use_z_filtering:
                visible_node_ids.add(node_id)
                if node_type == "room": visible_room_ids.add(node_id)
            else:
                if abs(center[2] - self.current_z) <= self.z_threshold:
                    visible_node_ids.add(node_id)
                    if node_type == "room": visible_room_ids.add(node_id)

        # dependency check
        if self.use_z_filtering and floor_candidates:
            for u, v, attr, in edges_data:
                if u in visible_room_ids and v in floor_candidates: visible_node_ids.add(v)
                elif v in visible_room_ids and u in floor_candidates: visible_node_ids.add(u)
        
        manual_hierarchy_nodes = {"room", "floor", "building", "city"}

        # process nodes
        for node_id, attr in nodes_data:
            if node_id not in visible_node_ids:
                continue
        
            # get raw center data
            raw_center = self._ensure_3d(attr.get("center", [0,0,0]))

            # apply visual offset
            node_type = attr.get("type", "unknown").lower().strip()
            offset = self.viz_center_offsets.get(node_type, np.array([0,0,0]))

            # fix mature graph visualization
            # if a mature graph is passed, the nodes would visualize at the wrong z position
            if node_type in ["building", "city"]:
                center = np.array([
                    raw_center[0] + offset[0],
                    raw_center[1] + offset[1],
                    offset[2]
                ])
            else:
                # get new center from raw data + visual offset
                center = raw_center + offset
            
            # update lookups (for interaction)
            self.node_coords[node_id] = center  # essential for update_selection()
            node_id_to_idx[node_id] = len(self.node_ids_list)
            self.node_ids_list.append(node_id)
            temp_coords_list.append(center)

            # styling
            node_type = attr.get("type", "Point")
            fmt = attr.get("viz_feat", "ko")
            color = _mpl_color_from_feat(fmt)

            marker = "o"
            possible_markers = {".", "o", "v", "^", "s", "*", "+", "x"}
            for char in fmt:
                if char in possible_markers:
                    marker = char
                    break

            if node_type == "ws": color = "red"
            size = 50 if node_type in manual_hierarchy_nodes else 30

            # add to scatter batch
            batch_key = (marker, size)
            if batch_key not in scatter_batches:
                scatter_batches[batch_key] = {"points": [], "colors": []}
            
            scatter_batches[batch_key]["points"].append(center)
            scatter_batches[batch_key]["colors"].append(color)

            # add legend
            if node_type not in legend_handles:
                h = self.ax.scatter([], [], [], marker=marker, color=color, label=node_type)
                legend_handles[node_type] = h

            # handle spcific viz types
            if attr["viz_type"] == "Line":
                viz_data = np.array(attr["viz_data"])
                if viz_data.shape[1] == 2:
                    viz_data = np.hstack([viz_data, np.zeros((viz_data.shape[0], 1))])
                
                line_fmt = attr.get("viz_feat", "k-")
                line_color = _mpl_color_from_feat(line_fmt)
                linewidth = attr.get("linewidth", 1.5)
                
                self.ax.plot(viz_data[:,0], viz_data[:,1], viz_data[:,2], c=line_color, linewidth=linewidth)
                
                if self.show_normals:
                    normal_3d = self._ensure_3d(attr.get("normal", [0,0,1]))
                    norm_line = np.stack([center, center + normal_3d/4])
                    self.ax.plot(norm_line[:,0], norm_line[:,1], norm_line[:,2], "b", linewidth=linewidth)

            if self.show_node_centers:
                # add center dot to batches
                center_key = (".", 10)  # 10 approx markersize=2 
                if center_key not in scatter_batches: 
                    scatter_batches[center_key] = {"points": [], "colors": []}
                
                scatter_batches[center_key]["points"].append(raw_center) 
                scatter_batches[center_key]["colors"].append("red") # manually add color

            if self.include_node_ids:
                self.ax.text(center[0], center[1], center[2], str(node_id), fontsize=9, color='black')

        # convert coords to numpy array for vectorized hover
        if temp_coords_list:
            self.node_coords_array = np.vstack(temp_coords_list)
        else:
            self.node_coords_array = np.empty((0, 3))

        # draw node batches
        for (marker, size), data in scatter_batches.items():
            pts = np.array(data["points"])
            colors = data["colors"] # This is a list of color strings/tuples
            
            # one draw call for all colors sharing this marker/size
            self.ax.scatter(pts[:,0], pts[:,1], pts[:,2], 
                            c=colors, marker=marker, s=size, depthshade=False)

        # process edges
        for edge_data in edges_data:
            node_id1, node_id2, attr = edge_data

            # visibility check
            if node_id1 not in visible_node_ids or node_id2 not in visible_node_ids:
                continue

            idx1 = node_id_to_idx[node_id1]
            idx2 = node_id_to_idx[node_id2]

            self.edge_indices.append((idx1, idx2))
            self.edge_keys.append((node_id1, node_id2))

            is_working = attr.get("working_edge", False)

            if is_working:
                if not self.show_working_edges:
                    continue
            else:
                if not self.show_edges:
                    continue

            # add to collection segments
            p1 = temp_coords_list[idx1]
            p2 = temp_coords_list[idx2]
            segments.append([p1, p2])

            fmt = attr.get("viz_feat", "k-")
            color = _mpl_color_from_feat(fmt)
            edge_colors.append(color)

            # compiled legend for all edges into the "common" label
            edge_label = "common"

            if edge_label not in legend_handles:
                h, = self.ax.plot([], [], [], color="k", linewidth=1.5, alpha=1.0, label=edge_label)
                legend_handles[edge_label] = h

            ### gets too cluttered after adding all edges ###
            # dynamic legend for edges
            # edge_label = attr.get("type", "edge")
            # if edge_label not in legend_handles:
            #    h, = self.ax.plot([], [], [], color=color, linewidth=1.5, label=edge_label)
            #    legend_handles[edge_label] = h
        
        # draw edge batch
        if segments:
            lc = Line3DCollection(segments, colors=edge_colors, linewidths=1.5, alpha=1.0)
            self.ax.add_collection(lc)

        # restore cam state
        if saved_cam_state:
            self.ax.set_xlim3d(saved_cam_state["xlim"])
            self.ax.set_ylim3d(saved_cam_state["ylim"])
            self.ax.set_zlim3d(saved_cam_state["zlim"])
            self.ax.view_init(elev=saved_cam_state["elev"], azim=saved_cam_state["azim"])
            self.ax.set_box_aspect([1, 1, 1])
        else:
            # aspect ratio logic (only on first load) -> matplotlib auto is very inefficient
            if self.node_coords_array.shape[0] > 0:
                max_xyz = np.max(self.node_coords_array, axis=0)
                min_xyz = np.min(self.node_coords_array, axis=0)
                
                mid_xyz = (max_xyz + min_xyz) * 0.5
                max_range = np.max(max_xyz - min_xyz)
                padding = max_range * 0.1
                half_span = (max_range + padding) * 0.5
                
                self.ax.set_xlim(mid_xyz[0] - half_span, mid_xyz[0] + half_span)
                self.ax.set_ylim(mid_xyz[1] - half_span, mid_xyz[1] + half_span)
                self.ax.set_zlim(mid_xyz[2] - half_span, mid_xyz[2] + half_span)
                
                self.ax.set_box_aspect([1, 1, 1])

        # add legend
        if legend_handles:
            self.ax.legend(handles=list(legend_handles.values()), loc="upper right")

        # controls display
        controls_text = (
            "CONTROLS:\n"
            "=========================\n"
            "L-Click    : Select Node\n"
            "Shift+R   : Create Room (from Selection)\n"
            "Shift+E   : Create Edge (between 2 nodes)\n"
            "Shift+F   : Create Floor\n"
            "Shift+B  : Create Building\n"
            "Shift+C  : Create City\n"
            "Del         : Delete Selection\n"
            "Shift+U  : Force Node Position recalculation\n"
            "Shift+S  : Save Graph\n"
            "=========================\n"
            "w : Working Edges\n"
            "m : WS Nodes (Planes)\n"
            "c : Debug Centers\n"
            "z : Toggle Z-Filter\n"
            "h : Toggle Control Display"
        )

        if self.show_controls:
            # text2D places text in screen coordinates (0,0 is bottom-left, 1,1 is top-right)
            self.ax.text2D(
                0.02, 0.20,             # X=2%, Y=20% (bottom left corner)
                controls_text,
                transform=self.fig.transFigure,  # Anchors text to the window, not the 3D graph
                verticalalignment='top',
                horizontalalignment='left',
                fontsize=9,
                color='black',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
            )
        else:
            self.ax.text2D(
                0.02, 0.05,             # X=2%, Y=5% (bottom left corner)
                "h : Toggle Control Display",
                transform=self.fig.transFigure,  # Anchors text to the window, not the 3D graph
                verticalalignment='top',
                horizontalalignment='left',
                fontsize=9,
                color='black',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
            )

        # init hover artist for highlights
        self.hover_artist = self.ax.scatter(
            [], [], [],
            c="orange",
            s=140,
            alpha=0.8,
            edgecolors="red",
            linewidth=2.0,
            zorder=150
        )

        # update title (including z-level)
        z_level_str = f"Group: {self.current_group_type}"
        if self.use_z_filtering:
            z_level_str += f" | Z-Slice: {self.current_z:.2f}m"
        else:
            z_level_str += f" | Z-Filter: OFF"
        self.ax.set_title(z_level_str)

        self.update_selection()
        self.fig.canvas.draw_idle()
        self._has_view = True

    def update_selection(self):
        """Updates or creates persistent scatter overlays for each active group."""
        for grp_type in list(self.selection_artists.keys()):
            if grp_type in self.selection_artists:
                try:
                    self.selection_artists[grp_type].remove()
                except Exception:
                    pass
                del self.selection_artists[grp_type]
        
        for grp_type, grp in self.active_groups.items():
            if not grp:
                continue

            offsets = [self.node_coords[nid] for nid in grp if nid in self.node_coords]
            if not offsets:
                continue

            offsets = np.array(offsets)

            scatter = self.ax.scatter(
                offsets[:,0], offsets[:,1], offsets[:,2],
                s=150, facecolors='none',
                edgecolors=self.group_colors.get(grp_type, "red"),
                linewidths=2, depthshade=False
            )

            self.selection_artists[grp_type] = scatter

    def on_hover(self, event):
        """
        On hover, highlight node and connected edges if they are within a certain distance.
        using vectorized distance calculation for efficiency.
        """
        if event.inaxes != self.ax: return
        if event.button is not None: return 

        if self.node_coords_array is None or len(self.node_coords_array) == 0:
            return

        mouse_pos = np.array([event.x, event.y])
        
        # vectorized projection
        screen_coords = self._project_coords()
        
        # vec dist to nodes
        deltas = screen_coords - mouse_pos
        dists_sq = np.sum(deltas**2, axis=1)
        min_node_idx = np.argmin(dists_sq)
        min_node_dist = np.sqrt(dists_sq[min_node_idx])
        
        closest_node_id = self.node_ids_list[min_node_idx] if min_node_dist < 20.0 else None
        
        # vec dist to edges
        closest_edge = None
        min_edge_dist = float("inf")
        
        if closest_node_id is None and len(self.edge_indices) > 0:
            # create endpoints for all edges in screenspace
            edge_idxs = np.array(self.edge_indices)
            
            p1s = screen_coords[edge_idxs[:, 0]]
            p2s = screen_coords[edge_idxs[:, 1]]
            
            # pt to seg distance vec
            seg_vecs = p2s - p1s
            pt_vecs = mouse_pos - p1s 
            
            seg_lens_sq = np.sum(seg_vecs**2, axis=1)
            
            # dot products
            dots = np.sum(pt_vecs * seg_vecs, axis=1)
            
            # t param, handle zero length
            t = np.zeros_like(dots)
            mask = seg_lens_sq > 0
            t[mask] = dots[mask] / seg_lens_sq[mask]
            t = np.clip(t, 0.0, 1.0)
            
            #  closest points
            closest_pts = p1s + seg_vecs * t[:, np.newaxis]
            
            # dists
            edge_dists_sq = np.sum((mouse_pos - closest_pts)**2, axis=1)
            min_edge_idx = np.argmin(edge_dists_sq)
            min_edge_dist = np.sqrt(edge_dists_sq[min_edge_idx])
            
            if min_edge_dist < 30.0:
                u, v = self.edge_keys[min_edge_idx]
                # Need p1, p2 in 3D for visualization
                p1_3d = self.node_coords_array[edge_idxs[min_edge_idx, 0]]
                p2_3d = self.node_coords_array[edge_idxs[min_edge_idx, 1]]
                closest_edge = {"u": u, "v": v, "p1": p1_3d, "p2": p2_3d}

        # decision logic
        target_type = None 
        node_valid = closest_node_id is not None
        edge_valid = closest_edge is not None
        
        if node_valid and not edge_valid:
            target_type = "node"
        elif edge_valid and not node_valid:
            target_type = "edge"
        elif node_valid and edge_valid:
            if min_node_dist < (min_edge_dist + 5.0):
                target_type = "node"
            else:
                target_type = "edge"

        # update tracker
        if target_type == "edge":
            self.hovered_edge = (closest_edge['u'], closest_edge['v'])
        else:
            self.hovered_edge = None

        # check state change
        current_hover_signature = (target_type, closest_node_id if target_type == "node" else self.hovered_edge)
        
        if hasattr(self, '_last_hover_sig') and self._last_hover_sig == current_hover_signature:
            return 
        self._last_hover_sig = current_hover_signature
        
        # clear highlights
        self._clear_highlights()
        
        if target_type == "node":
            # lookup 3D coord
            coord = self.node_coords_array[min_node_idx]
            self.hover_artist._offsets3d = ([coord[0]], [coord[1]], [coord[2]])
            
            # iterate edges since we have edge_keys
            neighbor_lines = []
            u_edges = [k for k in self.edge_keys if k[0] == closest_node_id]
            v_edges = [k for k in self.edge_keys if k[1] == closest_node_id]
            
            for (u, v) in u_edges:
                # v is neighbor
                if v in self.node_ids_list:
                    idx = self.node_ids_list.index(v)
                    neighbor_lines.append(self.node_coords_array[idx])
            for (u, v) in v_edges:
                # u is neighbor
                if u in self.node_ids_list:
                    idx = self.node_ids_list.index(u)
                    neighbor_lines.append(self.node_coords_array[idx])
            
            for n_coord in neighbor_lines:
                self._draw_highlight_line(coord, n_coord, color="purple", linewidth=2.0)

        elif target_type == "edge":
            p1, p2 = closest_edge['p1'], closest_edge['p2']
            self._draw_highlight_line(p1, p2, color="yellow", linewidth=4.0)
            self.hover_artist._offsets3d = ([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])
        
        else:
            self.hover_artist._offsets3d = ([], [], [])

        self.fig.canvas.draw_idle()
        
    def on_click(self, event):
        """Uses 3D projection for hit detection and toggles node selection."""
        if event.inaxes != self.ax:
            return
        
        screen_coords = self._project_coords()
        if len(screen_coords) == 0:
            return
        
        click_pos = np.array([event.x, event.y])

        # fast distance check
        deltas = screen_coords - click_pos
        dists_sq = np.sum(deltas**2, axis=1)
        min_idx = np.argmin(dists_sq)
        min_dist = np.sqrt(dists_sq[min_idx])
        
        closest_id = self.node_ids_list[min_idx]

        if closest_id is None:
            return
            
        if min_dist > 30.0:
            if self.logger:
                self.logger.info(f"Click too far: {min_dist:.2f}px")
            return

        # Toggle Selection Logic (Same as before)
        if self.current_group_type not in self.active_groups:
            self.active_groups[self.current_group_type] = set()
            
        if closest_id in self.active_groups[self.current_group_type]:
            self.active_groups[self.current_group_type].remove(closest_id)
            if self.logger: self.logger.info(f"Node {closest_id} removed.")
        else:
            self.active_groups[self.current_group_type].add(closest_id)
            if self.logger: self.logger.info(f"Node {closest_id} added.")

        self.update_selection()
        self.fig.canvas.draw_idle()

    def generate_new_node_id(self, graph):
        """Generate a new node Id such that new nodes dont overlate with existing ones"""
        node_ids = list(graph.get_nodes_ids())
        if not node_ids:
            return 0
        
        if all(isinstance(n, int) for n in node_ids):
            return max(node_ids) + 1
        
        existing = set(str(n) for n in node_ids)
        i = 0
        while True:
            candidate = f"n_{i}"
            if candidate not in existing:
                return candidate
            i += 1
    
    def compute_center_from_nodes(self, graph, node_ids):
        """Computes the center of selected nodes to place the newly created node at said center"""
        centers = []
        for nid in node_ids:
            attrs = graph.get_attributes_of_node(nid)
            if "center" in attrs:
                centers.append(np.array(attrs["center"], dtype=float))
        if centers:
            return np.mean(np.vstack(centers), axis=0)
        
        # fallback
        return np.array([0.0, 0.0, 0.0])
    
    def create_room_from_planes(self, full_graph, plane_ids, group_type="R"):
        """
        Create a room node connected to selected plane nodes,
        and attach it to floor -> building -> city nodes
        returns: new room node id
        """
        plane_ids = list(plane_ids)
        if not plane_ids:
            return None

        # room center = mean of plane centers
        room_center = self.compute_center_from_nodes(full_graph, plane_ids)
        room_center[2] = 0.0

        offset = self.viz_center_offsets["room"]
        viz_center = room_center + offset
        viz_feat = self.sdg_colors["room"]

        rid = self.generate_new_node_id(full_graph)
        room_attr = {
            "type": "room",
            "center": room_center,
            "viz": {
                "type": "Point",
                "center": viz_center,
                "feat": viz_feat,
                "linewidth": 1,
                "alpha": 1.0, 
                "size": 1
            },
            "viz_type": "Point",
            "viz_data": viz_center,
            "viz_feat": viz_feat,
            "group_type": group_type,
        }
        full_graph.add_nodes([(rid, room_attr)])

        # connect planes → room
        plane_room_edges = []
        for pid in plane_ids: 
            plane_room_edges.append((pid, rid, {
                "type": "ws_belongs_room",
                "viz_feat": "k-",
                "linewidth": 1.0,
                "alpha": 0.7,
                "working_edge": True,
            }))
        full_graph.add_edges(plane_room_edges)

        # update visualized graph
        if self.graph is not None:
            try:
                viz_node_ids = set(self.graph.get_nodes_ids())
            except Exception:
                viz_node_ids = set()

            if all(pid in viz_node_ids for pid in plane_ids):
                self.graph.add_nodes([(rid, room_attr)])
                self.graph.add_edges(plane_room_edges)
                self.draw_graph(preserve_view=True)

        return rid
    
    def delete_selection(self):
        """
        Deletes selected
        - Acitve Node (red circle) + all connected edges
        - highlighted edge (yellow)
        """
        if self.full_graph is None:
            return
        
        # collect all nodes currently selected across all group types
        ids_to_delete = []
        for nodes in self.active_groups.values():
            ids_to_delete.extend(list(nodes))
        
        if ids_to_delete:
            # remove from graph
            self.full_graph.remove_nodes(ids_to_delete)

            # remove from visualization 
            if self.graph != self.full_graph:
                try:
                    self.graph.remove_nodes(ids_to_delete)
                except:
                    pass
            
            if self.logger:
                self.logger.info(f"Deleted nodes: {ids_to_delete}")

            # recalculate all node positions after node deletion
            self._safe_recalc(self.full_graph)
            if self.graph != self.full_graph:
                self._safe_recalc(self.graph)

            # clear selection and redraw
            self.active_groups = {}

            # clear visuals
            self.update_selection()
            self.draw_graph(preserve_view=True)
            return
    
        if self.hovered_edge:
            u, v = self.hovered_edge

            try:
                self.full_graph.remove_edges([(u, v)])
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to delete edge: {e}")

            # update visual
            if self.graph != self.full_graph:
                try:
                    self.graph.remove_edges([(u, v)])
                except:
                    pass

            if self.logger:
                self.logger.info(f"Deleted edge between {u} and {v}")

            # recalculate all node positions after edge deletion
            self._safe_recalc(self.full_graph)
            if self.graph != self.full_graph:
                self._safe_recalc(self.graph)

            # clear hover state
            self.hovered_edge = None
            self._clear_highlights()
            self.draw_graph(preserve_view=True)
            return
        
        if not ids_to_delete:
            if self.logger:
                self.logger.info("No nodes selected to delete.")
            return

    def create_edge_between_selected(self):
        """
        Creates an edge between exactly two selected nodes (no features).
        """
        if self.full_graph is None:
            if self.logger:
                self.logger.error("No graph available")
            return

        selected_ids = list(self.active_groups.get(self.current_group_type, set()))

        if self.logger:
            self.logger.info(f"Shift+E pressed. Selected nodes in group '{self.current_group_type}': {selected_ids}")

        if len(selected_ids) != 2:
            if self.logger:
                self.logger.info(f"Please select exactly 2 nodes. Currently selected: {len(selected_ids)}")
            return

        node_a, node_b = selected_ids

        edge = [(node_a, node_b, {
            "type": "manual_edge",
            "viz_feat": "b-",
            "linewidth": 1.0,
            "alpha": 0.8,
        })]

        try:
            self.full_graph.add_edges(edge)

            if self.graph is not None and self.graph != self.full_graph:
                try:
                    viz_node_ids = set(self.graph.get_nodes_ids())
                    if node_a in viz_node_ids and node_b in viz_node_ids:
                        self.graph.add_edges(edge)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Could not add edge to visualization graph: {e}")

            if self.logger:
                self.logger.info(f"Created edge between nodes {node_a} and {node_b}")

            self.active_groups[self.current_group_type] = set()
            self.update_selection()
            self.draw_graph(preserve_view=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating edge: {e}")
            import traceback
            traceback.print_exc()

    def manual_create_hierarchy_node(self, node_type):
        """
        Creates any type node connecting to the currently selected node
        
        Args:
        - node_type (str): "floor", "building", "city"
        """
        # get selected ids
        selected_ids = []
        for nodes in self.active_groups.values():
            selected_ids.extend(list(nodes))

        if not selected_ids:
            if self.logger:
                self.logger.info(f"Select nodes first to create a {node_type}")
                return
            
        offset_vec = self.viz_center_offsets.get(node_type, np.array([0,0,0]))
        viz_feat = self.sdg_colors.get(node_type, "k")

        edge_type = "unknown_hierarchy"
        if node_type == "wall": edge_type = "ws_belongs_wall"
        elif node_type == "floor": edge_type = "room_belongs_floor"
        elif node_type == "building": edge_type = "floor_belongs_building"
        elif node_type == "city": edge_type = "building_belongs_city"
        
        # calc center
        child_centers = []
        for nid in selected_ids:
            attr = self.full_graph.get_attributes_of_node(nid)
            c = np.array(attr.get("center", [0,0,0]), dtype=float)
            if len(c) < 3:
                c = np.pad(c, (0, 3-len(c)))
            child_centers.append(c)
                
        if not child_centers:
            return
        
        new_center = np.mean(np.vstack(child_centers), axis=0)
        final_center = new_center.copy()

        # apply offset
        if node_type in ["building", "city"]:
            final_center[2] = 0.0

        viz_center = final_center + offset_vec

        # create node
        new_id = self.generate_new_node_id(self.full_graph)

        # Note: flat viz_ attributes will be cleaned on save
        node_attr = {
            "type": node_type,
            "center": final_center,
            "viz": {
                "type": "Point",
                "data": viz_center,
                "feat": viz_feat
            },
            "viz_type": "Point",
            "viz_data": viz_center,
            "viz_feat": viz_feat
        }

        self.full_graph.add_nodes([(new_id, node_attr)])
        
        # create edges
        # Note: working_edge will be cleaned on save
        new_edges = []
        for cid in selected_ids:
            new_edges.append((cid, new_id, {
                "type": edge_type,
                "viz_feat": "k-",
                "linewidth": 1.0,
                "alpha": 0.6,
                "working_edge": True
            }))

        self.full_graph.add_edges(new_edges)

        if self.graph != self.full_graph:
            self.graph.add_nodes([(new_id, node_attr)])
            self.graph.add_edges(new_edges)

        if self.logger:
            self.logger.info(f"Created {node_type} {new_id} connected to {len(selected_ids)} nodes.")
        
        # clear selection
        self.active_groups = {}
        self.draw_graph(preserve_view=True)

    def on_key(self, event):
        """Handles key events to change group type or finalize the active group."""
        # Visual toggles to avoid clutter on denser graphs
        if event.key == "e":  # toggle edges
            self.show_edges = not self.show_edges
            if self.logger:
                self.logger.info(f"Toggled edges to {self.show_edges}")
            self.draw_graph(preserve_view=True)
            return
        if event.key == "n":  # toggle normals
            self.show_normals = not self.show_normals
            if self.logger:
                self.logger.info(f"Toggled normals to {self.show_normals}")
            self.draw_graph(preserve_view=True)
            return
        if event.key == "c":  # toggle node centers
            self.show_node_centers = not self.show_node_centers
            if self.logger:
                self.logger.info(f"Toggled centers to {self.show_node_centers}")
            self.draw_graph(preserve_view=True)
            return
        if event.key == "w":  # toggle working edges
            self.show_working_edges = not self.show_working_edges
            if self.logger:
                self.logger.info(f"Toggled working edges to {self.show_working_edges}")
            self.draw_graph(preserve_view=True)
            return
        if event.key == "i":  # toggle node ids
            self.include_node_ids = not self.include_node_ids
            if self.logger:
                self.logger.info(f"Toggled node ids to {self.include_node_ids}")
            self.draw_graph(preserve_view=True)
            return
        if event.key == "h":  # toggle control help
            self.show_controls = not self.show_controls
            if self.logger:
                self.logger.info(f"Toggled control display to {self.show_controls}")
            self.draw_graph(preserve_view=True)
            return

        # create edge between selected nodes
        if event.key == "E":  # Shift + e
            self.create_edge_between_selected()
            return

        # manual hierarchy creation
        if event.key == "W":  # Shift + w
            self.manual_create_hierarchy_node("wall")
            if self.logger:
                self.logger.info("Created node of type: wall")
        if event.key == "F":  # Shift + f to avoid fullscreen keybind (f)
            self.manual_create_hierarchy_node("floor")
            if self.logger:
                self.logger.info("Created node of type: floor")
            return
        if event.key == "B":  # Shift + b 
            self.manual_create_hierarchy_node("building")
            if self.logger:
                self.logger.info("Created node of type: building")
            return
        if event.key == "C":  # Shift + c
            self.manual_create_hierarchy_node("city")
            if self.logger:
                self.logger.info("Created node of type: city")
            return
        if event.key == "delete":  # del
            self.delete_selection()
            if self.logger:
                self.logger.info(f"Deleted selected node and corresponding edges")
            return
        if event.key == "U":  # Shift + u for update (recalc hierarchy positions)
            if self.full_graph:
                if self.logger:
                    self.logger.info("Recalculating hierarchy positions...")
                self._safe_recalc(self.full_graph)
                if self.graph != self.full_graph and self.graph is not None:
                    self._safe_recalc(self.graph)
                self.draw_graph(preserve_view=True)
                return
        
        # save graph 
        if event.key == "S":  # Shift + s (s in matplotlib is screenshot)
            self.save_graph()
            if self.logger:
                self.logger.info(f"Saving graph...")
            return
        
        # Navigation keys for z levels
        if event.key == "z":  # Toggle Filter Mode
            self.use_z_filtering = not self.use_z_filtering
            if self.logger:
                self.logger.info(f"Z-Filtering set to {self.use_z_filtering}")
            self.draw_graph(preserve_view=True)
            return
        if event.key == "up":  # Move Up a floor
            if not self.use_z_filtering:
                self.use_z_filtering = True
                if self.logger:
                    self.logger.info("Z-Filtering Enabled via Navigation")
            else:
                self.current_z += 3 # floor height 
                if self.logger: self.logger.info(f"Moved UP to Z={self.current_z}")
            self.draw_graph(preserve_view=True)
            return
        if event.key == "down":  # Move Down a floor
            if not self.use_z_filtering:
                self.use_z_filtering = True
                if self.logger:
                    self.logger.info("Z-Filtering Enabled via Navigation")
            else:
                self.current_z -= 3
                if self.logger: self.logger.info(f"Moved DOWN to Z={self.current_z}")
            self.draw_graph(preserve_view=True)
            return

        # previous key functions
        if event.key in self.group_colors:
            self.current_group_type = event.key
            if self.logger:
                self.logger.info(f"Current group type set to '{self.current_group_type}'.")
        if event.key == "R":  # Shift + r to finalize group and create room from selection
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
                self.create_room_from_planes(self.full_graph, finalized, self.current_group_type)  # added to enable working on mature graphs
                # create_room_from_planes already triggers a redraw, hence the lines below are commented out
                # self.update_selection()
                # self.fig.canvas.draw_idle()
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

    def _compute_default_save_path(self):
        """Create a default filename based on time"""
        ts = int(time.time() * 1000)
        fname = f"Interactive_graph_autosave_{ts}.pkl"

        if self.default_save_dir is not None:
            os.makedirs(self.default_save_dir, exist_ok=True)
            return os.path.join(self.default_save_dir, fname)
        return fname
    
    def save_graph(self, path=None):
        """
        Cleans the graph of interactive attributes (is_working, flat viz_type, etc.)
        syncs nested viz dicts to match SDG format
        saves grpah as a pickled GW object
        """
        source_data = self.full_graph if self.full_graph is not None else self.graph

        if source_data is None:
            if self.logger:
                self.logger.warning("No full_graph set, nothing to save")
            return
        
        if path is None:
            if self.current_save_path is None:
                raw_name = ""
                if self.has_qt and self.QtInputDialog:
                    text, ok = self.QtInputDialog.getText(None, "Save Graph", "Enter filename: ")
                    if ok and text:
                        raw_name = text.strip()
                else:
                    print("\n" + "="*40)
                    try:
                        raw_name = input(">>> Enter name for this graph\n>>> ").strip()
                    except EOFError:
                        raw_name = "autosave_graph"

                if not raw_name:
                    raw_name = self._compute_default_save_path()
                    print(f"No name entered. Defaulting to: {raw_name}")

                # ensure file name extension
                if not raw_name.endswith(".pkl"):
                    raw_name += ".pkl"

                # apply dir
                if self.default_save_dir is not None:
                    os.makedirs(self.default_save_dir, exist_ok=True)
                    self.current_save_path = os.path.join(self.default_save_dir, raw_name)
                else:
                    self.current_save_path = raw_name

            # use established path
            path = self.current_save_path
        
        if self.logger:
            self.logger.info("Cleaning Graph for export...")

        # check if it is already a GW
        is_wrapper = hasattr(source_data, "clone") and hasattr(source_data, "get_attributes_of_all_nodes")

        if is_wrapper:
            clean_graph = source_data.clone()  # from GraphWrapper
        else:
            if self.logger:
                self.logger.info("Input was not a GW. Converting now...")

            try:
                clean_graph = GraphWrapper(graph_obj=copy.deepcopy(source_data))
            except Exception as e:
                if self.logger: self.logger.warning(f"Could not convert graph to GW: {e}")

        keys_to_clean_node = ["viz_type", "viz_data", "viz_feat", "group_type", "active"]
        # keys_to_clean_edge = ["working_edge", "viz_feat", "linewidth", "alpha"]

        # clean excessive node info
        for nid, attrs in clean_graph.get_attributes_of_all_nodes():
            if "viz" not in attrs:
                attrs["viz"] = {}

            if "center" in attrs and "viz" in attrs:
                ntype = attrs.get("type")
                geom_center = np.array(attrs["center"])
                offset = self.viz_center_offsets.get(ntype, np.array([0,0,0]))

                attrs["viz"]["center"] = geom_center + offset
                attrs["viz"]["type"] = attrs.get("viz_type", "Point")

                if "feat" not in attrs["viz"] and "viz_feat" in attrs:
                    attrs["viz"]["feat"] = attrs["viz_feat"]

            for key in keys_to_clean_node:
                if key in attrs:
                    del attrs[key]
            
            clean_graph.update_node_attrs(nid, attrs)

        for u, v, attrs in clean_graph.get_attributes_of_all_edges():
            if "working_edge" in attrs:
                del attrs["working_edge"]

            clean_graph.update_edge_attrs((u, v), attrs)

        # save
        try:
            with open(path, "wb") as f:
                pickle.dump(clean_graph, f, protocol=pickle.HIGHEST_PROTOCOL)  # save the cleaned graph

            if self.logger:
                self.logger.info(f"[INFO]: Graph saved to {path}")
            else:
                print(f"Graph saved to {path}")  # still display a message even if no logger

        except Exception as e:
            if self.logger:
                self.logger.warning(f"[WARNING]: Error saving graph to {path}: {e}")
            else:
                print(f"Error saving graph to {path}: {e}")

    def show(self):
        """Displays the interactive graph window (blocking call)."""
        plt.show(block=True)

    def get_finalized_groups(self):
        return self.finalized_groups