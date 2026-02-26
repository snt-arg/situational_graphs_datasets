#!/usr/bin/env python3
"""
Load a fixed list of .plkl/.pkl files from a given folder (non-recursive) and
print per file:
- number of graphs
- average degree across graphs (mean of per-graph avg node degree)

Python: 3.8.10
"""

import argparse
import pickle
from pathlib import Path

import networkx as nx


# EDIT THIS LIST: filenames (just names, no paths)
PICKLE_FILES = [
    'ssg_manhhattan_small_3000.pickle',
    'ssg_L_small_3000.pkl',
    'ssg_L_small_buildings_2f2b.pkl',
    'msd_small_3467.pkl',
    'msd_floors_2f_200o.pkl',
    'msd_buildings_1085_2f2b_150o.pkl',
    'ssg_real.pkl',
    'ssg_real_floors.pkl',
    'ssg_real_buildings.pkl',
]


def load_pickle(path):
    with path.open("rb") as f:
        return pickle.load(f)


def is_nx_graph(obj):
    return isinstance(obj, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))


def coerce_to_graph_list(obj, path):
    if is_nx_graph(obj):
        return [obj]

    if isinstance(obj, (list, tuple)):
        if all(is_nx_graph(g) for g in obj):
            return list(obj)
        raise TypeError(
            "{}: loaded a list/tuple but not all elements are NetworkX graphs".format(path.name)
        )

    if isinstance(obj, dict):
        for k in ("graphs", "nx_graphs", "data", "items"):
            if k in obj and isinstance(obj[k], (list, tuple)) and all(is_nx_graph(g) for g in obj[k]):
                return list(obj[k])
        raise TypeError(
            "{}: loaded a dict but couldn't find a list of graphs under common keys".format(path.name)
        )

    raise TypeError("{}: unsupported pickle content type: {}".format(path.name, type(obj).__name__))


def avg_degree_of_graph(G):
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    deg_sum = sum(d for _, d in G.degree())
    return float(deg_sum) / float(n)


def mean_over_graphs(graphs, fn):
    graphs = list(graphs)
    if not graphs:
        return 0.0
    return sum(fn(G) for G in graphs) / float(len(graphs))


def fmt_num(x, decimals=2):
    return "{:.{d}f}".format(x, d=decimals)


def print_table(rows):
    headers = ["File", "#graphs", "avg_nodes", "avg_edges", "avg_degree"]
    # Build string rows (so we can compute column widths)
    srows = []
    for r in rows:
        srows.append([
            r["file"],
            str(r["n_graphs"]),
            fmt_num(r["avg_nodes"], 2),
            fmt_num(r["avg_edges"], 2),
            fmt_num(r["avg_degree"], 4),
        ])

    # Column widths
    cols = list(zip(headers, *srows)) if srows else [headers]
    widths = [max(len(str(cell)) for cell in col) for col in cols]

    def line(sep="-", junction="+"):
        return junction + junction.join(sep * (w + 2) for w in widths) + junction

    def row(cells):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"

    print(line("-", "+"))
    print(row(headers))
    print(line("=", "+"))
    for sr in srows:
        print(row(sr))
    print(line("-", "+"))


def main(folder):

    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit("Not a folder: {}".format(folder))

    if not PICKLE_FILES:
        raise SystemExit("PICKLE_FILES is empty. Edit the script and add filenames to it.")

    rows = []

    for fname in PICKLE_FILES:
        p = folder / fname
        if not p.exists():
            rows.append({
                "file": fname,
                "n_graphs": 0,
                "avg_nodes": float("nan"),
                "avg_edges": float("nan"),
                "avg_degree": float("nan"),
            })
            continue

        obj = load_pickle(p)
        graphs = coerce_to_graph_list(obj, p)

        rows.append({
            "file": fname,
            "n_graphs": len(graphs),
            "avg_nodes": mean_over_graphs(graphs, lambda G: G.number_of_nodes()),
            "avg_edges": mean_over_graphs(graphs, lambda G: G.number_of_edges()),
            "avg_degree": mean_over_graphs(graphs, avg_degree_of_graph),
        })

    print_table(rows)



if __name__ == "__main__":
    main("/home/adminpc/workspaces/reasoning_ws/src/situational_graphs_datasets/datasets/ifh")
