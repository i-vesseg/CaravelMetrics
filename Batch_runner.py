# -*- coding: utf-8 -*-
"""
Batch_runner.py

Batch entry mode for vessel-level metrics: BINARY mask (one per subject,
already reduced to a graph via Graph_extractor.extract_graph) + a list of
endpoint coordinates (one per vessel of interest, LPS+ world mm -- see
Path_analyzer.py's module docstring for the coordinate frame). Multi-label
masks are not used anywhere in this module: each vessel is defined purely by
an ostium->endpoint path on the graph (Path_analyzer.path_to_point), not by
mask membership (Vessel_aggregator.nodes_in_mask / metrics_per_label).

Nothing in Graph_extractor.py, Segment_extractor.py, Metric_extractor.py is
modified. Path_analyzer.path_to_point and Vessel_aggregator.aggregate_path_metrics
are reused as-is; this module only adds the per-subject graph loading and the
per-row loop.

Input CSV format (one row per vessel of interest):
    subject_id, vessel_name, x, y, z
e.g.:
    CCmaxi_4275, LAD, 12.3, -45.6, 78.9

`x, y, z` must be in the same LPS+ world-mm frame as G.nodes[n]['pos']
(see Path_analyzer.py's module docstring for how to obtain such a point,
e.g. by picking on the Segment_visualizer.export_segments_vtp() output in
3D Slicer).
"""
import logging
import os
import pickle

import pandas as pd

from Graph_extractor import extract_graph
from coronary_extension.Segment_extractor import build_segments_merged
from coronary_extension.Path_analyzer import path_to_point
from coronary_extension.Vessel_aggregator import aggregate_path_metrics

logger = logging.getLogger(__name__)

ENDPOINT_CSV_COLUMNS = ['subject_id', 'vessel_name', 'x', 'y', 'z']


def _load_subject_graph(subject_id, nifti_root, output_root):
    """
    Load (or build, once) the graph/segments for one subject, reusing an
    existing vessel_data.pkl if present -- same reuse pattern as
    run_demo.py's STEP 1. Tries f"{subject_id}.nii" then f"{subject_id}.nii.gz"
    under `nifti_root` if the pickle doesn't already exist.

    Segments come from build_segments_merged, not the raw build_segments, so
    degenerate segments are merged away. G_clean (not the raw graph) is
    returned so callers stay consistent with edge_to_seg -- see
    Segment_extractor.build_segments_merged's docstring on why passing the
    raw graph here would risk a KeyError in Path_analyzer.path_to_point.

    Returns (G_clean, node_radius_map, segments, edge_to_seg).
    """
    output_folder = os.path.join(output_root, subject_id)
    pkl_path = os.path.join(output_folder, "vessel_data.pkl")

    if os.path.exists(pkl_path):
        logger.info("Subject %s: reusing existing pickle %s", subject_id, pkl_path)
    else:
        nifti_path = os.path.join(nifti_root, f"{subject_id}.nii")
        if not os.path.exists(nifti_path):
            nifti_path = os.path.join(nifti_root, f"{subject_id}.nii.gz")
        if not os.path.exists(nifti_path):
            raise FileNotFoundError(
                f"No nifti found for subject {subject_id!r} under {nifti_root!r} "
                f"(tried .nii and .nii.gz)."
            )
        os.makedirs(output_folder, exist_ok=True)
        extract_graph(nifti_path, output_folder, save_skeleton_mask=False)

    with open(pkl_path, 'rb') as f:
        package = pickle.load(f)
    G = package['graph']
    node_radius_map = package['node_radius_map']

    segments, edge_to_seg, node_to_segs, G_clean = build_segments_merged(G)
    return G_clean, node_radius_map, segments, edge_to_seg


def run_batch(endpoint_csv_path, nifti_root, output_root, results_csv_path):
    """
    Run the batch path-based aggregation described in the module docstring.

    Parameters:
        endpoint_csv_path : str
            CSV with columns ENDPOINT_CSV_COLUMNS (see module docstring).
        nifti_root : str
            Folder containing one nifti per subject_id (f"{subject_id}.nii[.gz]"),
            used only if that subject's vessel_data.pkl doesn't already exist
            under output_root.
        output_root : str
            Folder containing (or to receive) one subfolder per subject_id,
            each with a vessel_data.pkl (Graph_extractor.extract_graph's output).
        results_csv_path : str
            Where to write the one-row-per-vessel results table.

    A subject whose graph can't be loaded/built, or a row whose point fails
    to snap/path (e.g. an out-of-mask coordinate, or a point in a connected
    component with no degree-1 ostium candidate), is logged and recorded with
    an 'error' column instead of stopping the whole batch.

    Returns the results DataFrame (also written to results_csv_path).
    """
    endpoints_df = pd.read_csv(endpoint_csv_path)
    missing_cols = set(ENDPOINT_CSV_COLUMNS) - set(endpoints_df.columns)
    if missing_cols:
        raise ValueError(f"Endpoint CSV is missing required column(s): {sorted(missing_cols)}")

    rows = []
    for subject_id, group in endpoints_df.groupby('subject_id'):
        try:
            G, node_radius_map, segments, edge_to_seg = _load_subject_graph(
                subject_id, nifti_root, output_root
            )
        except Exception as e:
            logger.error("Subject %s: could not load/build graph -- %s", subject_id, e)
            for _, row in group.iterrows():
                rows.append({
                    'subject_id': subject_id, 'vessel_name': row['vessel_name'],
                    'x': row['x'], 'y': row['y'], 'z': row['z'],
                    'error': f"graph load/build failed: {e}",
                })
            continue

        for _, row in group.iterrows():
            vessel_name = row['vessel_name']
            point = (float(row['x']), float(row['y']), float(row['z']))
            try:
                result = path_to_point(G, node_radius_map, segments, edge_to_seg, point)
                agg = aggregate_path_metrics(result, G=G)
                rows.append({
                    'subject_id': subject_id,
                    'vessel_name': vessel_name,
                    'x': point[0], 'y': point[1], 'z': point[2],
                    'ostium_node': result['ostium_node'],
                    'ostium_criterion': result['ostium_criterion'],
                    'error': None,
                    **agg,
                })
            except Exception as e:
                logger.error(
                    "Subject %s, vessel %s: path aggregation failed -- %s",
                    subject_id, vessel_name, e
                )
                rows.append({
                    'subject_id': subject_id, 'vessel_name': vessel_name,
                    'x': point[0], 'y': point[1], 'z': point[2],
                    'error': str(e),
                })

    results_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(results_csv_path) or '.', exist_ok=True)
    results_df.to_csv(results_csv_path, index=False)
    logger.info("Wrote %d row(s) to %s", len(results_df), results_csv_path)
    return results_df


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Batch ostium->endpoint vessel aggregation.")
    parser.add_argument('endpoint_csv', help="CSV with columns: subject_id, vessel_name, x, y, z")
    parser.add_argument('--nifti-root', default=os.path.join("data_test", "Eva_Coronaries"))
    parser.add_argument('--output-root', default="Results")
    parser.add_argument('--results-csv', default=os.path.join("Results", "batch_vessel_metrics.csv"))
    args = parser.parse_args()

    df = run_batch(args.endpoint_csv, args.nifti_root, args.output_root, args.results_csv)
    print(f"\n{len(df)} row(s) written to {args.results_csv}")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df.to_string(index=False))
