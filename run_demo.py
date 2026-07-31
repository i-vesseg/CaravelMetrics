# -*- coding: utf-8 -*-
"""
run_demo.py

End-to-end demo of the additive pipeline (Segment_extractor, Path_analyzer,
Vessel_aggregator, Segment_visualizer) on top of the unmodified baseline
(Graph_extractor.extract_graph), run once on a real subject
(CCmaxi_4275, chosen because it's a genuine full-vessel tree in
data_test/Eva_Coronaries/ AND has a matching single-segment label file --
same shape/affine -- in data_test/, needed for the aggregation step).

Nothing in Graph_extractor.py, Metric_extractor.py, Segment_extractor.py,
Vessel_aggregator.py, Segment_visualizer.py or CaravelMetrics.py is
modified. This script only imports and calls them as-is.
"""
import os
import pickle

import networkx as nx
import numpy as np
import nibabel as nib

from Graph_extractor import extract_graph
from coronary_extension.Segment_extractor import build_segments, build_segments_merged, segments_to_dataframe
from coronary_extension.Path_analyzer import path_to_point, find_default_ostium, path_chain_to_dataframe
from coronary_extension.Vessel_aggregator import metrics_per_label
from coronary_extension.Segment_visualizer import export_segments_vtp

SUBJECT = "CCmaxi_4275"
NIFTI_PATH = os.path.join("data_test", "Eva_Coronaries", f"{SUBJECT}.nii")
SINGLE_SEGMENT_MASK_PATH = os.path.join("data_test", "CCmaxi_4275_1-Segment_1-label.nii.gz")
OUTPUT_FOLDER = os.path.join("Results", SUBJECT)


def main():
    summary = {}

    print("=" * 70)
    print("STEP 1 -- generate (or reuse) the pickle")
    print("=" * 70)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    pkl_path = os.path.join(OUTPUT_FOLDER, "vessel_data.pkl")
    if os.path.exists(pkl_path):
        print(f"Reusing existing pickle: {pkl_path}")
    else:
        extract_graph(NIFTI_PATH, OUTPUT_FOLDER, save_skeleton_mask=False)

    with open(pkl_path, "rb") as f:
        package = pickle.load(f)
    G = package["graph"]
    node_radius_map = package["node_radius_map"]

    print(f"\nGraph size: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Pickle dict keys: {list(package.keys())}")
    summary["subject"] = SUBJECT
    summary["pkl_path"] = pkl_path

    print("\n" + "=" * 70)
    print("STEP 2 -- build the merged partition and prove the relaxed invariants")
    print("=" * 70)
    raw_segments, _, _ = build_segments(G)
    n_degenerate = sum(1 for s in raw_segments if s["degenerate"])
    total_graph_length = G.size(weight="weight")

    degenerate_length = sum(
        sum(G[u][v]["weight"] for u, v in zip(s["nodes"][:-1], s["nodes"][1:]))
        for s in raw_segments if s["degenerate"]
    )
    degenerate_length_frac = degenerate_length / total_graph_length if total_graph_length > 0 else float("nan")

    print(f"Raw segments (build_segments): {len(raw_segments)} total, {n_degenerate} degenerate")
    print(f"Degenerate segments' share of total graph length: "
          f"{degenerate_length:.3f} / {total_graph_length:.3f} = {degenerate_length_frac:.4%}")

    segments, edge_to_seg, node_to_segs, G_clean = build_segments_merged(G)
    print(f"Merged segments (build_segments_merged): {len(segments)} total (0 degenerate, by construction)")

    # Relaxed invariants: the degenerate segments' own edges/length are
    # intentionally NOT covered by any merged segment -- that loss is by
    # design (the discarded stubs/bridges themselves). Everything else
    # (merged segments + skipped pure cycles) must still account for the
    # rest of the raw graph exactly.
    topo = {n for n in G if G.degree(n) != 2}
    covered_edges = sum(len(s["nodes"]) - 1 for s in segments)
    removed_degenerate_edges = sum(len(s["nodes"]) - 1 for s in raw_segments if s["degenerate"])
    skipped_edges = sum(
        G.subgraph(c).number_of_edges()
        for c in nx.connected_components(G)
        if not (c & topo)
    )
    partition_edges_ok = (covered_edges + removed_degenerate_edges + skipped_edges == G.number_of_edges())
    partition_index_ok = (len(edge_to_seg) == covered_edges)

    total_segment_length = sum(
        G[u][v]["weight"]
        for s in segments
        for u, v in zip(s["nodes"][:-1], s["nodes"][1:])
    )
    skipped_length = sum(
        G.subgraph(c).size(weight="weight")
        for c in nx.connected_components(G)
        if not (c & topo)
    )
    length_diff = abs((total_segment_length + degenerate_length + skipped_length) - total_graph_length)
    partition_length_ok = length_diff < 1e-6

    print(f"Edges: merged={covered_edges} + removed_degenerate={removed_degenerate_edges} + "
          f"skipped(pure cycles)={skipped_edges}, total graph edges: {G.number_of_edges()} -> "
          f"{'OK' if partition_edges_ok else 'MISMATCH'}")
    print(f"edge_to_seg index size == covered merged edges: {len(edge_to_seg)} == {covered_edges} -> "
          f"{'OK' if partition_index_ok else 'MISMATCH'}")
    print(f"Length: merged={total_segment_length:.6f} + removed_degenerate={degenerate_length:.6f} + "
          f"skipped={skipped_length:.6f} = {total_segment_length + degenerate_length + skipped_length:.6f} "
          f"vs {total_graph_length:.6f} (diff={length_diff:.2e}) -> "
          f"{'OK' if partition_length_ok else 'MISMATCH'}")

    if not (partition_edges_ok and partition_index_ok and partition_length_ok):
        print("\nPARTITION INVARIANT FAILED -- stopping here.")
        return

    summary["partition_check"] = "OK"
    summary["n_segments"] = len(segments)
    summary["n_degenerate_removed"] = n_degenerate
    summary["degenerate_length_frac"] = degenerate_length_frac

    print("\n" + "=" * 70)
    print("STEP 3 -- the path query (ostium -> farthest endpoint)")
    print("=" * 70)
    largest_component = max(nx.connected_components(G_clean), key=len)
    ostium_node = find_default_ostium(G_clean, node_radius_map, largest_component)
    print(f"Default ostium node: {ostium_node}")

    endpoints = [n for n in largest_component if G_clean.degree(n) == 1 and n != ostium_node]
    lengths = nx.single_source_dijkstra_path_length(G_clean.subgraph(largest_component), ostium_node, weight="weight")
    farthest_endpoint = max(endpoints, key=lambda n: lengths.get(n, -1.0))
    direct_path_length = lengths[farthest_endpoint]
    print(f"Farthest endpoint: {farthest_endpoint} (direct shortest-path length = {direct_path_length:.3f} mm)")

    query_point = G_clean.nodes[farthest_endpoint]["pos"]
    result = path_to_point(G_clean, node_radius_map, segments, edge_to_seg, query_point, ostium=ostium_node)

    print(f"\nOrdered chain of elementary segments, ostium -> farthest endpoint "
          f"({len(result['segment_chain'])} segment(s)):")
    header = f"{'seg_id':>6} {'dir/partial':<11} {'length':>9} {'cum_length':>11} {'mean_curv':>10} {'arc/chord':>10}"
    print(header)
    print("-" * len(header))
    cum_length = 0.0
    for entry in result["segment_chain"]:
        cum_length += entry["length"]
        m = entry["metrics"]
        mean_curv = m["spline_mean_curvature"] if m is not None else float("nan")
        arc_over_chord = m["arc_over_chord"] if m is not None else float("nan")
        tag = entry["direction"] if entry["full_segment"] else "partial"
        print(f"{entry['seg_id']:>6} {tag:<11} {entry['length']:>9.3f} {cum_length:>11.3f} "
              f"{mean_curv:>10.4f} {arc_over_chord:>10.4f}")

    chain_length = sum(entry["length"] for entry in result["segment_chain"])
    length_diff = abs(chain_length - direct_path_length)
    path_check_ok = length_diff < 1e-6
    print(f"\nChain length sum: {chain_length:.6f} mm vs direct path length: {direct_path_length:.6f} mm "
          f"(diff={length_diff:.2e}) -> {'OK' if path_check_ok else 'MISMATCH'}")
    assert path_check_ok, "Segment chain length does not match the direct ostium->endpoint path length!"

    summary["path_query_assert"] = "OK" if path_check_ok else "FAILED"
    summary["path_query_length"] = chain_length

    print("\n" + "=" * 70)
    print("STEP 4 -- vessel aggregation (single-segment mask, if grid/affine match)")
    print("=" * 70)
    if not os.path.exists(SINGLE_SEGMENT_MASK_PATH):
        print(f"No single-segment mask found at {SINGLE_SEGMENT_MASK_PATH}; skipping.")
        summary["aggregation"] = "skipped (no mask file)"
    else:
        ref_img = nib.load(NIFTI_PATH)
        mask_img = nib.load(SINGLE_SEGMENT_MASK_PATH)
        same_shape = (mask_img.shape == ref_img.shape)
        same_affine = np.allclose(mask_img.affine, ref_img.affine)
        print(f"Reference image ({NIFTI_PATH}): shape={ref_img.shape}")
        print(f"Mask image      ({SINGLE_SEGMENT_MASK_PATH}): shape={mask_img.shape}")
        print(f"Same shape: {same_shape}, same affine (np.allclose): {same_affine}")

        if not (same_shape and same_affine):
            print("Shape/affine mismatch -- NOT running aggregation (would silently mislabel nodes). Skipping.")
            summary["aggregation"] = "skipped (shape/affine mismatch)"
        else:
            rows = metrics_per_label(G_clean, mask_img, segments, edge_to_seg, node_to_segs)
            print(f"\nPer-vessel result ({len(rows)} label(s)):")
            for row in rows:
                excl_frac = row['curv_excluded_length_frac']
                excl_frac_str = f"{excl_frac:.4f}" if excl_frac is not None else "n/a"
                print(f"  label={row['label']}: n_elementary={row['n_elementary']}, "
                      f"n_pruned={row['n_pruned']}, length={row['length']:.3f} mm, "
                      f"arc_over_chord={row['arc_over_chord']:.4f}, "
                      f"mean_curvature={row['mean_curvature']:.4f}, "
                      f"curv_excluded_length_frac={excl_frac_str}, "
                      f"non_simple_path={row['non_simple_path']}")
            summary["aggregation"] = "ran"
            summary["aggregation_rows"] = rows

    print("\n" + "=" * 70)
    print("STEP 5 -- export for Slicer")
    print("=" * 70)
    vtp_path = os.path.join(OUTPUT_FOLDER, "vessel_segments_colored.vtp")
    csv_path = os.path.join(OUTPUT_FOLDER, "segments_table.csv")
    export_segments_vtp(segments, G_clean, vtp_path)
    df = segments_to_dataframe(segments, G_clean)
    df.to_csv(csv_path, index=False)
    print(f"Saved segments table CSV: {csv_path}")
    summary["vtp_path"] = vtp_path
    summary["csv_path"] = csv_path

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Subject: {summary['subject']}")
    print(f"Pickle: {summary['pkl_path']}")
    print(f"Partition check: {summary['partition_check']} "
          f"({summary['n_segments']} merged segments, {summary['n_degenerate_removed']} degenerate "
          f"removed/merged away, {summary['degenerate_length_frac']:.4%} of total length)")
    print(f"Path-query assert: {summary['path_query_assert']} (chain length = {summary['path_query_length']:.3f} mm)")
    print(f"Aggregation: {summary['aggregation']}")
    print(f"Slicer VTP: {summary['vtp_path']}")
    print(f"Slicer segments CSV: {summary['csv_path']}")


if __name__ == "__main__":
    main()
