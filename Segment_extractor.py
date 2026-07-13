# -*- coding: utf-8 -*-
"""
Segment_extractor.py

Adds a new, alternative "segment" definition on top of the vessel graph
produced by Graph_extractor.extract_graph(). Nothing in Graph_extractor.py or
Metric_extractor.py is modified: compute_tortuosity_metrics is only imported
and called as-is.

Existing definition (Metric_extractor.extract_segments_from_component_using_shortest_path):
    segment = shortest path root -> endpoint. These paths overlap (an edge
    can belong to several root->endpoint paths), which is why `counts` /
    division by n(s) exists in compute_tortuosity_metrics.

This module's definition:
    segment = maximal chain of nodes between two "topological" nodes
    (degree != 2), with no topological node in its interior:
        degree 1   -> endpoint
        degree 2   -> pass-through node (segment interior)
        degree >=3 -> bifurcation
    Every edge of G belongs to exactly one segment (a partition), so no
    `counts` is needed when calling compute_tortuosity_metrics.

Reused from the existing codebase (see Step 0 inspection):
    - Graph_extractor.extract_graph(): builds G as an undirected
      networkx.Graph, node attribute 'pos' (3D world-space coordinate,
      np.ndarray shape (3,)), edge attribute 'weight' (Euclidean length
      between the two endpoints' 'pos').
    - Metric_extractor.compute_tortuosity_metrics(points, smoothing=0,
      n_samples=500, counts=None): fits a cubic B-spline to `points` and
      returns spline_arc_length, spline_chord_length,
      spline_mean_curvature, spline_mean_square_curvature,
      spline_rms_curvature, arc_over_chord, fit_rmse. counts=None is
      equivalent to weighting every point by 1 (i.e. no counts).
    - No coord -> node snap helper (KDTree/closest_node) exists elsewhere
      in the codebase, and none is needed here: build_segments works
      purely on graph topology, not on world coordinates.
"""
import logging

import networkx as nx
import numpy as np
import pandas as pd

from Metric_extractor import compute_tortuosity_metrics

logger = logging.getLogger(__name__)

# Segments with fewer nodes than this are flagged 'degenerate' and get
# metrics=None instead of being passed to compute_tortuosity_metrics.
# Set to 4, not 3: compute_tortuosity_metrics fits a CUBIC B-spline
# (Metric_extractor.py: `if pts.shape[0] < 4` -> returns an all-NaN dict),
# which needs at least 4 points to be fit at all. A 3-node segment would
# pass the old threshold as non-degenerate only to come back out as NaN;
# flagging it degenerate up front avoids ever calling the spline fit on
# too few points.
N_MIN = 4


def build_segments(G):
    """
    Partition an undirected vessel graph G into topological segments.

    A segment is a maximal chain of nodes running between two topological
    nodes (degree != 2 -- endpoints or bifurcations), with no topological
    node in its interior. Every edge of G belongs to exactly one segment.

    G is walked as a whole rather than per connected component: starting
    from every topological node's incident edges and skipping edges already
    walked handles multiple connected components naturally in one pass.

    A connected component that is a pure cycle (no node of degree != 2,
    e.g. an isolated loop/orphan artifact) has no topological node to start
    a walk from; such a component is logged as a WARNING and skipped (its
    edges do not appear in any segment).

    Parameters:
        G : networkx.Graph
            Undirected vessel graph with node attribute 'pos' and edge
            attribute 'weight', as produced by Graph_extractor.extract_graph.

    Returns:
        segments : list of dict, each with keys
            'id'         : int, index into `segments`
            'nodes'      : list of node ids in walk order (endpoints included)
            'endpoints'  : (nodes[0], nodes[-1])
            'degenerate' : bool, True if len(nodes) < N_MIN
            'metrics'    : dict from compute_tortuosity_metrics, or None if
                           degenerate
        edge_to_seg : dict
            frozenset({u, v}) -> segment id
        node_to_segs : dict
            node -> list of segment ids the node belongs to (a bifurcation
            belongs to several segments, which is expected)
    """
    topo = {n for n in G if G.degree(n) != 2}

    seen = set()
    segments = []

    for t in topo:
        for nb in G.neighbors(t):
            e0 = frozenset((t, nb))
            if e0 in seen:
                continue
            path = [t, nb]
            seen.add(e0)
            prev, cur = t, nb
            while cur not in topo:
                nxt = next(x for x in G.neighbors(cur) if x != prev)
                seen.add(frozenset((cur, nxt)))
                path.append(nxt)
                prev, cur = cur, nxt
            segments.append({
                'id': len(segments),
                'nodes': path,
                'endpoints': (path[0], path[-1]),
                'degenerate': len(path) < N_MIN,
            })

    for component in nx.connected_components(G):
        if not (component & topo):
            logger.warning(
                "Skipping pure-cycle component with no topological node "
                "(degree != 2): %d nodes, %d edges. Not included in any segment.",
                len(component), G.subgraph(component).number_of_edges()
            )

    edge_to_seg = {}
    node_to_segs = {}
    for seg in segments:
        nodes = seg['nodes']
        for n in nodes:
            node_to_segs.setdefault(n, []).append(seg['id'])
        for u, v in zip(nodes[:-1], nodes[1:]):
            edge_to_seg[frozenset((u, v))] = seg['id']

    for seg in segments:
        if seg['degenerate']:
            seg['metrics'] = None
            continue
        pts = np.array([G.nodes[n]['pos'] for n in seg['nodes']])
        seg['metrics'] = compute_tortuosity_metrics(pts)

    return segments, edge_to_seg, node_to_segs


def segments_to_dataframe(segments, G):
    """
    Build a one-row-per-segment table from build_segments() output.

    fit_rmse is intentionally NOT included: compute_tortuosity_metrics fits
    the spline with s=0 (an interpolating spline, passing exactly through
    every input point) and then computes fit_rmse by evaluating that spline
    at the SAME parameter values used to fit it (Metric_extractor.py,
    `spline_at_u = splev(u, tck)` vs the original `pts`). That residual is
    structurally ~0 for every segment regardless of node count or
    tortuosity, so the column carries no information here.

    Parameters:
        segments : list of dict, as returned by build_segments (first element)
        G : networkx.Graph, the same graph passed to build_segments

    Returns:
        pandas.DataFrame with one row per segment.
    """
    metric_cols = [
        'spline_arc_length', 'spline_chord_length', 'spline_mean_curvature',
        'spline_mean_square_curvature', 'spline_rms_curvature', 'arc_over_chord',
    ]
    float_cols = ['length'] + metric_cols

    rows = []
    for seg in segments:
        nodes = seg['nodes']
        length = sum(G[u][v]['weight'] for u, v in zip(nodes[:-1], nodes[1:]))
        row = {
            'id': int(seg['id']),
            'n_nodes': int(len(nodes)),
            'endpoint_a': int(seg['endpoints'][0]),
            'endpoint_b': int(seg['endpoints'][1]),
            'degenerate': bool(seg['degenerate']),
            'length': float(length),
        }
        metrics = seg['metrics']
        for key in metric_cols:
            row[key] = None if metrics is None else float(metrics[key])
        rows.append(row)

    df = pd.DataFrame(
        rows,
        columns=['id', 'n_nodes', 'endpoint_a', 'endpoint_b', 'degenerate', 'length'] + metric_cols,
    )
    df[float_cols] = df[float_cols].round(4)
    return df


if __name__ == "__main__":
    # --- Sanity-check block on one Eva_Coronaries subject (print only, no CSV writing) ---
    # The only deviation from existing code: pointing the input at data_test/Eva_Coronaries
    # and using a scratch output folder outside the repo for the intermediate vessel_data.pkl
    # (extract_graph itself is called unmodified).
    import os
    import pickle
    import tempfile

    from Graph_extractor import extract_graph

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    nifti_path = os.path.join("data_test", "Eva_Coronaries", "CCmaxi_14907.nii")
    scratch_dir = os.path.join(tempfile.gettempdir(), "segment_extractor_sanity_check")
    os.makedirs(scratch_dir, exist_ok=True)

    pkl_path = os.path.join(scratch_dir, "vessel_data.pkl")
    if not os.path.exists(pkl_path):
        extract_graph(nifti_path, scratch_dir, save_skeleton_mask=False)
    else:
        print(f"Reusing existing extracted graph at {pkl_path}")

    with open(pkl_path, 'rb') as f:
        package = pickle.load(f)
    G = package['graph']

    print(f"\nLoaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    segments, edge_to_seg, node_to_segs = build_segments(G)

    n_degenerate = sum(1 for s in segments if s['degenerate'])
    print(f"\nSegments found: {len(segments)} ({n_degenerate} degenerate, N_MIN={N_MIN})")

    # --- Partition check: every edge belongs to exactly one segment ---
    covered_edges = sum(len(s['nodes']) - 1 for s in segments)
    skipped_edges = sum(
        G.subgraph(c).number_of_edges()
        for c in nx.connected_components(G)
        if not (c & {n for n in G if G.degree(n) != 2})
    )
    print(f"Edges covered by segments: {covered_edges}")
    print(f"Edges in skipped pure-cycle components: {skipped_edges}")
    print(f"Total graph edges: {G.number_of_edges()}")
    assert covered_edges + skipped_edges == G.number_of_edges(), \
        "Edges are not exactly partitioned by segments (+ skipped pure cycles)!"
    assert len(edge_to_seg) == covered_edges, "edge_to_seg index size mismatch!"
    print("OK: every edge of G belongs to exactly one segment (modulo skipped pure cycles).")

    # --- Length check: sum of segment lengths ~= total graph length ---
    total_graph_length = G.size(weight='weight')
    total_segment_length = sum(
        G[u][v]['weight']
        for s in segments
        for u, v in zip(s['nodes'][:-1], s['nodes'][1:])
    )
    skipped_length = sum(
        G.subgraph(c).size(weight='weight')
        for c in nx.connected_components(G)
        if not (c & {n for n in G if G.degree(n) != 2})
    )
    print(f"\nTotal graph length (weight sum): {total_graph_length:.3f}")
    print(f"Total segment length:            {total_segment_length:.3f}")
    print(f"Skipped pure-cycle length:        {skipped_length:.3f}")
    assert abs((total_segment_length + skipped_length) - total_graph_length) < 1e-6, \
        "Segment length sum does not match total graph length!"
    print("OK: segment lengths sum to the total graph length (modulo skipped pure cycles).")

    # --- Print a few sample segments ---
    print("\nSample segments:")
    for s in segments[:3]:
        m = s['metrics']
        m_str = "degenerate (no metrics)" if m is None else {k: round(v, 4) if v is not None and not (isinstance(v, float) and np.isnan(v)) else v for k, v in m.items()}
        print(f"  seg {s['id']}: endpoints={s['endpoints']}, n_nodes={len(s['nodes'])}, metrics={m_str}")

    # --- Full segments table (one row per segment) + CSV export ---
    df = segments_to_dataframe(segments, G)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    print("\nFull segments table:")
    print(df.to_string(index=False))

    csv_path = os.path.join(scratch_dir, "segments_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved segments table CSV: {csv_path}")
