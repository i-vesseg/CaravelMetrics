# -*- coding: utf-8 -*-
"""
Path_analyzer.py

Answers: "given a 3D point, what is the path from the coronary ostium to
that point, as an ordered chain of elementary segments, with metrics?"

Nothing in Graph_extractor.py, Metric_extractor.py, Segment_extractor.py,
Vessel_aggregator.py or CaravelMetrics.py is modified. This module only
imports and calls them as-is.

Why this is a query, not a new segment definition
---------------------------------------------------
Segment_extractor.build_segments(G) partitions G's edges into elementary
segments (maximal degree!=2 -to- degree!=2 chains): a basis, each edge owned
by exactly one segment. A root-to-point path is a *composition* of that
basis, not a basis element itself -- it generally crosses several elementary
segments, entering and leaving each one at a bifurcation. This module
answers the path query by:
  1) computing the shortest path (by edge weight) from an ostium node to a
     node close to the query point, on the raw graph G, then
  2) walking that path's edges through `edge_to_seg` (returned by
     build_segments but unused elsewhere -- see Vessel_aggregator.py's
     `metrics_per_label` docstring, which reserves it for exactly this) to
     regroup them into the ordered chain of elementary segments crossed.

Because the ostium is always a topological node (degree 1) and every
elementary segment can only be entered/left at a topological node, every
segment in the chain is crossed in FULL except possibly the last one --
the path stops wherever the query point was snapped to, which may be a
degree-2 node in a segment's interior. For a fully-crossed, non-degenerate
segment this module reuses that segment's already-computed `metrics`
(build_segments already ran compute_tortuosity_metrics on it -- no need to
refit the spline). Only a partial trailing segment gets a fresh
compute_tortuosity_metrics call on the actually-traversed sub-chain.

Coordinate frame
-----------------
`query_point` (and `ostium`, if given as a point) must be in the same frame
as G.nodes[n]['pos'], i.e. LPS+ world mm (see Graph_extractor.SLICER_MATRIX
and Vessel_aggregator.py's module docstring). This is exactly the frame
used by Segment_visualizer.export_segments_vtp()'s output: a point picked
by clicking on that model in 3D Slicer (File > Add Data, then read off the
click coordinates via a vtkCellPicker in Slicer's Python console) is
already in this frame, no transform needed.

Reused, not modified: Segment_extractor.build_segments,
Metric_extractor.compute_tortuosity_metrics.
"""
import os
import sys

# See Segment_extractor.py's module top for why this is needed: this file
# lives in coronary_extension/, one level below the repo root (baseline
# CaravelMetrics modules + sibling coronary_extension/ modules), and neither
# is a pip-installed package -- this puts both on sys.path so the flat
# imports below resolve whether this module is imported from a root script
# or run directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import logging

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from Metric_extractor import compute_tortuosity_metrics
from Segment_extractor import N_MIN

logger = logging.getLogger(__name__)


def _snap_point_to_node(G, point, node_ids=None):
    """
    Nearest graph node (by Euclidean distance in G's 'pos' frame) to `point`.

    Parameters:
        G : networkx.Graph
        point : array-like, shape (3,)
        node_ids : optional iterable of node ids to restrict the search to
            (e.g. a single connected component). Defaults to all of G.

    Returns:
        (node_id, distance) : nearest node and its distance to `point`.
    """
    ids = list(node_ids) if node_ids is not None else list(G.nodes())
    if not ids:
        raise ValueError("No candidate nodes to snap to.")
    positions = np.array([G.nodes[n]['pos'] for n in ids])
    tree = cKDTree(positions)
    dist, idx = tree.query(np.asarray(point, dtype=float))
    return ids[idx], float(dist)


def find_default_ostium(G, node_radius_map, node_ids):
    """
    Default ostium node within `node_ids`: the degree-1 node (endpoint) with
    the largest diameter (2 * radius). Same criterion as
    Metric_extractor.ROOT_CRITERIA[0] ('largest_endpoint'), the closest
    existing convention for "root" in this codebase -- reused here rather
    than inventing a new one.

    Raises ValueError if `node_ids` contains no degree-1 node.
    """
    endpoints = [n for n in node_ids if G.degree(n) == 1]
    if not endpoints:
        raise ValueError(
            "No degree-1 node (endpoint) found to use as a default ostium "
            "in this connected component; pass `ostium` explicitly."
        )
    return max(endpoints, key=lambda n: node_radius_map.get(n, 0.5))


def _resolve_ostium(G, node_radius_map, node_ids, ostium):
    """
    Resolve `ostium` (see path_to_point) to a node id already inside
    `node_ids`, plus a human-readable string describing how it was chosen.
    """
    if ostium is None:
        node = find_default_ostium(G, node_radius_map, node_ids)
        return node, 'largest_endpoint_in_component (default)'

    if isinstance(ostium, (int, np.integer)) or ostium in G:
        if ostium not in node_ids:
            raise ValueError(
                f"ostium node {ostium!r} is not in the same connected "
                f"component as the query point."
            )
        return ostium, 'user_specified_node'

    node, dist = _snap_point_to_node(G, ostium, node_ids)
    logger.info("Snapped user-specified ostium point to node %s (%.3f mm away).", node, dist)
    return node, 'user_specified_point'


def _segment_chain_from_path(path_nodes, edge_to_seg):
    """
    Group the edges of an ordered node path into the ordered chain of
    elementary segments they cross, using `edge_to_seg` (build_segments'
    edge -> segment-id index).

    Returns: list of (seg_id, chunk_nodes) in traversal order, where
    consecutive path edges that map to the same segment id are merged into
    one chunk. chunk_nodes is the sub-list of path_nodes actually spanned
    (endpoints included, shared between consecutive chunks).
    """
    chain = []
    current_seg = None
    current_nodes = []
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        seg_id = edge_to_seg[frozenset((u, v))]
        if seg_id != current_seg:
            if current_nodes:
                chain.append((current_seg, current_nodes))
            current_seg = seg_id
            current_nodes = [u, v]
        else:
            current_nodes.append(v)
    if current_nodes:
        chain.append((current_seg, current_nodes))
    return chain


def _chunk_metrics(chunk_nodes, seg_id, segments, G):
    """
    Build one segment_chain entry (see path_to_point) for a single
    (seg_id, chunk_nodes) pair produced by _segment_chain_from_path.
    """
    full_segment = segments[seg_id]
    full_nodes = full_segment['nodes']
    is_full = (
        len(chunk_nodes) == len(full_nodes)
        and (chunk_nodes == full_nodes or chunk_nodes == full_nodes[::-1])
    )

    length = sum(
        G[u][v]['weight'] for u, v in zip(chunk_nodes[:-1], chunk_nodes[1:])
    )

    if is_full:
        direction = 'forward' if chunk_nodes == full_nodes else 'reverse'
        metrics = full_segment['metrics']
        degenerate = full_segment['degenerate']
    else:
        direction = 'partial'
        degenerate = len(chunk_nodes) < N_MIN
        if degenerate:
            metrics = None
        else:
            pts = np.array([G.nodes[n]['pos'] for n in chunk_nodes])
            metrics = compute_tortuosity_metrics(pts)

    return {
        'seg_id': seg_id,
        'nodes': chunk_nodes,
        'direction': direction,
        'full_segment': is_full,
        'degenerate': degenerate,
        'length': float(length),
        'metrics': metrics,
    }


def path_to_point(G, node_radius_map, segments, edge_to_seg, query_point,
                   ostium=None, weight='weight'):
    """
    Path from the coronary ostium to a 3D query point, as an ordered chain
    of elementary segments (Segment_extractor.build_segments) with metrics.

    Parameters:
        G : networkx.Graph
            As produced by Graph_extractor.extract_graph (node attr 'pos',
            edge attr 'weight').
        node_radius_map : dict
            node -> radius (mm), from the same vessel_data.pkl as G.
        segments, edge_to_seg : as returned by Segment_extractor.build_segments(G)
            (node_to_segs is not needed here).
        query_point : array-like, shape (3,)
            Target point, in the same frame as G.nodes[n]['pos'] (see
            module docstring). Snapped to the nearest graph node.
        ostium : optional
            One of:
              - None (default): auto-pick the largest-diameter degree-1
                node in the query point's connected component (see
                find_default_ostium).
              - an existing node id in G.
              - a 3D point (array-like, shape (3,)): snapped to the nearest
                node in the query point's connected component.
        weight : str, default 'weight'
            Edge attribute used for the shortest path (nx.shortest_path).
            The graph is a vessel skeleton and normally a tree, but if a
            connected component contains a loop, weight='weight' picks the
            geodesically shortest route rather than the fewest-edges one.

    Returns:
        dict with keys:
          'query_point'          : np.ndarray, the input point
          'target_node'          : node id the query point was snapped to
          'target_snap_distance' : float, distance (mm) from query_point
                                    to that node
          'ostium_node'          : node id used as the path's start
          'ostium_criterion'     : str, how `ostium_node` was chosen
          'path_nodes'           : list of node ids, ostium -> target, in order
          'path_length'          : float, sum of edge weights along path_nodes
          'path_metrics'         : dict from compute_tortuosity_metrics over
                                    the whole path as one curve, or None if
                                    path_nodes has fewer than
                                    Segment_extractor.N_MIN points
          'segment_chain'        : list of dicts, one per elementary segment
                                    crossed, in traversal order, each with
                                    keys 'seg_id', 'nodes', 'direction'
                                    ('forward'/'reverse'/'partial'),
                                    'full_segment' (bool), 'degenerate' (bool),
                                    'length', 'metrics' (dict or None)
    """
    target_node, target_dist = _snap_point_to_node(G, query_point)
    component = nx.node_connected_component(G, target_node)

    ostium_node, ostium_criterion = _resolve_ostium(G, node_radius_map, component, ostium)

    path_nodes = nx.shortest_path(G, ostium_node, target_node, weight=weight)
    path_length = sum(
        G[u][v]['weight'] for u, v in zip(path_nodes[:-1], path_nodes[1:])
    )

    if len(path_nodes) < N_MIN:
        path_metrics = None
    else:
        pts = np.array([G.nodes[n]['pos'] for n in path_nodes])
        path_metrics = compute_tortuosity_metrics(pts)

    chain = _segment_chain_from_path(path_nodes, edge_to_seg)
    segment_chain = [
        _chunk_metrics(chunk_nodes, seg_id, segments, G)
        for seg_id, chunk_nodes in chain
    ]

    return {
        'query_point': np.asarray(query_point, dtype=float),
        'target_node': target_node,
        'target_snap_distance': target_dist,
        'ostium_node': ostium_node,
        'ostium_criterion': ostium_criterion,
        'path_nodes': path_nodes,
        'path_length': float(path_length),
        'path_metrics': path_metrics,
        'segment_chain': segment_chain,
    }


def path_chain_to_dataframe(result):
    """
    One-row-per-crossed-segment table from a path_to_point() result, in
    traversal order (ostium -> target).
    """
    metric_cols = [
        'spline_arc_length', 'spline_chord_length', 'spline_mean_curvature',
        'spline_mean_square_curvature', 'spline_rms_curvature', 'arc_over_chord',
    ]
    float_cols = ['length'] + metric_cols

    rows = []
    for order, entry in enumerate(result['segment_chain']):
        row = {
            'order': order,
            'seg_id': int(entry['seg_id']),
            'direction': entry['direction'],
            'full_segment': bool(entry['full_segment']),
            'degenerate': bool(entry['degenerate']),
            'n_nodes': len(entry['nodes']),
            'length': float(entry['length']),
        }
        metrics = entry['metrics']
        for key in metric_cols:
            row[key] = None if metrics is None else float(metrics[key])
        rows.append(row)

    df = pd.DataFrame(
        rows,
        columns=['order', 'seg_id', 'direction', 'full_segment', 'degenerate', 'n_nodes', 'length'] + metric_cols,
    )
    df[float_cols] = df[float_cols].round(4)
    return df


if __name__ == "__main__":
    # --- Sanity-check block on the CCmaxi_4275 Eva_Coronaries subject
    # (print only, no CSV writing). Same extract-graph-then-reuse pattern as
    # Segment_extractor.py / Vessel_aggregator.py's own __main__ blocks.
    import os
    import pickle
    import tempfile

    from Graph_extractor import extract_graph
    from Segment_extractor import build_segments

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    nifti_path = os.path.join("data_test", "Eva_Coronaries", "CCmaxi_4275.nii")
    scratch_dir = os.path.join(tempfile.gettempdir(), "path_analyzer_sanity_check")
    os.makedirs(scratch_dir, exist_ok=True)

    pkl_path = os.path.join(scratch_dir, "vessel_data.pkl")
    if not os.path.exists(pkl_path):
        extract_graph(nifti_path, scratch_dir, save_skeleton_mask=False)
    else:
        print(f"Reusing existing extracted graph at {pkl_path}")

    with open(pkl_path, 'rb') as f:
        package = pickle.load(f)
    G = package['graph']
    node_radius_map = package['node_radius_map']

    print(f"\nLoaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    segments, edge_to_seg, node_to_segs = build_segments(G)
    print(f"build_segments: {len(segments)} elementary segments")

    # Pick a query point far from any node's exact position, to exercise
    # snapping: the position of some node deep in the largest component,
    # nudged off-lattice by a sub-voxel offset.
    largest_component = max(nx.connected_components(G), key=len)
    deep_node = sorted(largest_component)[len(largest_component) // 2]
    query_point = G.nodes[deep_node]['pos'] + np.array([0.37, -0.21, 0.15])

    result = path_to_point(G, node_radius_map, segments, edge_to_seg, query_point)

    print(f"\nQuery point: {result['query_point']}")
    print(f"Snapped to target node {result['target_node']} "
          f"({result['target_snap_distance']:.3f} mm away)")
    print(f"Ostium node: {result['ostium_node']} (criterion: {result['ostium_criterion']})")
    print(f"Path: {len(result['path_nodes'])} nodes, length={result['path_length']:.3f} mm")
    if result['path_metrics'] is not None:
        print(f"Whole-path arc_over_chord: {result['path_metrics']['arc_over_chord']:.4f}")

    # --- Consistency checks ---
    chain = result['segment_chain']
    print(f"\nSegment chain: {len(chain)} elementary segment(s) crossed")

    # 1) Chain length must equal path length.
    chain_length = sum(entry['length'] for entry in chain)
    assert abs(chain_length - result['path_length']) < 1e-6, \
        "Segment chain length does not match path length!"
    print(f"OK: segment chain length ({chain_length:.3f}) matches path length "
          f"({result['path_length']:.3f}).")

    # 2) All segments but possibly the last must be crossed in full.
    n_partial = sum(1 for entry in chain if not entry['full_segment'])
    assert n_partial <= 1 and (n_partial == 0 or not chain[-1]['full_segment']), \
        "Only the last segment in the chain may be partial!"
    print(f"OK: at most the last segment is partial ({n_partial} partial segment(s)).")

    # 3) Concatenating chain node lists (dropping duplicated shared nodes at
    #    chunk boundaries) must reproduce path_nodes exactly.
    rebuilt = [chain[0]['nodes'][0]]
    for entry in chain:
        rebuilt.extend(entry['nodes'][1:])
    assert rebuilt == result['path_nodes'], \
        "Concatenated segment chain does not reproduce the original path!"
    print("OK: concatenated segment chain reproduces the ostium -> target path exactly.")

    print("\nChain detail:")
    for entry in chain:
        m = entry['metrics']
        m_str = "None" if m is None else f"arc_over_chord={m['arc_over_chord']:.4f}"
        print(f"  seg {entry['seg_id']}: direction={entry['direction']}, "
              f"full_segment={entry['full_segment']}, n_nodes={len(entry['nodes'])}, "
              f"length={entry['length']:.3f} mm, metrics=({m_str})")

    df = path_chain_to_dataframe(result)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    print("\nSegment chain table:")
    print(df.to_string(index=False))
