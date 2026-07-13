# -*- coding: utf-8 -*-
"""
Vessel_aggregator.py

Aggregates elementary segments (Segment_extractor.build_segments) into whole
endpoint-to-endpoint vessels defined by a voxel mask (binary or multi-class),
and computes metrics for each whole vessel. Reuses the single global graph G
already built by Graph_extractor.extract_graph -- no graph is re-extracted
per mask.

Step 0 inspection (Graph_extractor.py) -- coordinate/affine convention:
  - nii = nib.load(nifti_path); affine = nii.affine. Per the NIfTI standard,
    this affine maps voxel index (i, j, k) -> RAS+ world mm coordinates
    (nibabel always exposes qform/sform-derived affines in RAS+).
  - skel_indices = np.argwhere(skeleton_mask) (voxel indices, array order);
    skel_points_world = apply_affine(affine, skel_indices) -> RAS+ world mm.
    Orphan points go through the same apply_affine(affine, ...) call.
  - G.nodes[n]['pos'] starts out as these RAS+ points, then is
    Laplacian-smoothed in place (still RAS+; smoothing doesn't change frame).
  - CRITICAL LAST STEP ("APPLY ALIGNMENT (SLICER MATRIX)", right before the
    pickle is saved): every node's pos is overwritten with
    `new_pos = SLICER_MATRIX[:3,:3] @ pos + SLICER_MATRIX[:3,3]`.
    SLICER_MATRIX = diag(-1, -1, 1) with zero translation, i.e. an
    RAS+ -> LPS+ flip (sign flip on x and y).
    => The 'pos' stored on G (what build_segments / compute_tortuosity_metrics
    operate on) is in LPS+ world mm, NOT the original nibabel RAS+ affine
    space, and NOT voxel index space.
  - Graph_extractor itself needs to invert this exact chain to map a node's
    pos back to a voxel index (for its radius lookup): it undoes the SLICER
    flip with `inv_slicer_rot = np.linalg.inv(rotation)` (self-inverse, since
    diag(-1,-1,1)^2 = I, translation is zero), then applies
    `apply_affine(inv_affine, ...)`. This module reuses that exact chain
    (importing SLICER_MATRIX rather than hardcoding a second copy of it),
    applied to a possibly different nifti (the mask), which is assumed to
    share the same voxel grid/affine as the segmentation originally used to
    build G.

Reused, not modified: Segment_extractor.build_segments,
Metric_extractor.compute_tortuosity_metrics, Graph_extractor.SLICER_MATRIX.
"""
import itertools
import logging

import networkx as nx
import numpy as np
from nibabel.affines import apply_affine

from Graph_extractor import SLICER_MATRIX
from Metric_extractor import compute_tortuosity_metrics

logger = logging.getLogger(__name__)

# A segment is kept in a vessel if a STRICT MAJORITY of its nodes fall inside
# the mask -- this avoids grabbing a neighbouring segment just because a
# shared bifurcation node happens to be inside the mask.
MEMBERSHIP_FRAC = 0.5


def _pos_to_voxel_index(pos_lps, inv_affine):
    """
    Invert the exact chain Graph_extractor.extract_graph uses to go from a
    node's stored 'pos' (LPS+ world mm, post-SLICER_MATRIX) back to a voxel
    index (i, j, k), for a nifti sharing the segmentation's grid/affine.
    """
    rot = SLICER_MATRIX[:3, :3]
    translation = SLICER_MATRIX[:3, 3]
    pos_ras = np.linalg.inv(rot) @ (pos_lps - translation)
    return apply_affine(inv_affine, pos_ras)


def nodes_in_mask(G, mask_img):
    """
    Test each node of G against a voxel mask (a nib image, loaded with the
    same nib.load()/get_fdata() pattern as Graph_extractor.extract_graph).

    Assumes mask_img shares the voxel grid/affine of the nifti originally
    used to build G (i.e. it is a labeling of the same volume, not an
    independently-registered image).

    Returns:
      - a set of node ids, if the mask is binary (at most one distinct
        nonzero value), or
      - a dict {label -> set of node ids}, if multi-class (more than one
        distinct nonzero value present).
    """
    data = mask_img.get_fdata()
    inv_affine = np.linalg.inv(mask_img.affine)
    shape = data.shape

    membership = {}
    for n in G.nodes():
        vox = _pos_to_voxel_index(G.nodes[n]['pos'], inv_affine)
        idx = np.round(vox).astype(int)
        if np.any(idx < 0) or np.any(idx >= shape):
            continue
        value = data[tuple(idx)]
        if value == 0:
            continue
        label = int(round(float(value)))
        membership.setdefault(label, set()).add(n)

    if len(membership) <= 1:
        return next(iter(membership.values())) if membership else set()
    return membership


def segments_in_vessel(node_set, segments, node_to_segs):
    """
    Collect the elementary segment ids (indices into `segments`) belonging to
    the vessel defined by `node_set`: a segment is kept if a strict majority
    (> MEMBERSHIP_FRAC) of its nodes are in node_set.
    """
    candidate_ids = set()
    for n in node_set:
        candidate_ids.update(node_to_segs.get(n, []))

    kept = []
    for seg_id in candidate_ids:
        nodes = segments[seg_id]['nodes']
        frac_in = sum(1 for n in nodes if n in node_set) / len(nodes)
        if frac_in > MEMBERSHIP_FRAC:
            kept.append(seg_id)
    return kept


def aggregate_vessel_metrics(seg_ids, segments, G):
    """
    Reassemble the whole vessel made of the elementary segments `seg_ids` and
    compute its metrics as a single unit. Elementary arc_over_chord values
    are NOT summed (non-additive): the whole-vessel spline is fit once on the
    concatenated, ordered path.

    Returns a dict: seg_ids, n_elementary, length (sum of the kept segments'
    edge weights), non_simple_path (True if the induced subgraph needed the
    longest-path fallback because it wasn't a simple chain), endpoints,
    metrics (the dict returned by compute_tortuosity_metrics).
    """
    if not seg_ids:
        return {
            'seg_ids': [], 'n_elementary': 0, 'length': 0.0,
            'non_simple_path': False, 'endpoints': None,
            'metrics': compute_tortuosity_metrics(np.zeros((0, 3))),
        }

    length = sum(
        G[u][v]['weight']
        for seg_id in seg_ids
        for u, v in zip(segments[seg_id]['nodes'][:-1], segments[seg_id]['nodes'][1:])
    )

    if len(seg_ids) == 1:
        seg = segments[seg_ids[0]]
        pts = np.array([G.nodes[n]['pos'] for n in seg['nodes']])
        metrics = seg['metrics'] if seg['metrics'] is not None else compute_tortuosity_metrics(pts)
        return {
            'seg_ids': seg_ids, 'n_elementary': 1, 'length': length,
            'non_simple_path': False, 'endpoints': seg['endpoints'], 'metrics': metrics,
        }

    nodes_union = set()
    for seg_id in seg_ids:
        nodes_union.update(segments[seg_id]['nodes'])
    subG = G.subgraph(nodes_union)

    degree_1_nodes = [n for n in subG.nodes() if subG.degree(n) == 1]
    non_simple_path = False

    if len(degree_1_nodes) == 2:
        ep1, ep2 = degree_1_nodes
    elif len(degree_1_nodes) > 2:
        non_simple_path = True
        logger.warning(
            "Vessel subgraph is not a simple path (%d endpoint-like nodes "
            "instead of 2 -- it contains internal bifurcations); falling "
            "back to the longest endpoint-to-endpoint path.",
            len(degree_1_nodes)
        )
        best = None
        for a, b in itertools.combinations(degree_1_nodes, 2):
            try:
                d = nx.shortest_path_length(subG, a, b, weight='weight')
            except nx.NetworkXNoPath:
                continue
            if best is None or d > best[0]:
                best = (d, a, b)
        if best is None:
            raise ValueError("Vessel subgraph has no path between any pair of its endpoints.")
        ep1, ep2 = best[1], best[2]
    else:
        raise ValueError(
            f"Vessel subgraph has {len(degree_1_nodes)} endpoint-like node(s); "
            "cannot define a vessel path (expected at least 2)."
        )

    ordered_nodes = nx.shortest_path(subG, ep1, ep2, weight='weight')
    pts = np.array([G.nodes[n]['pos'] for n in ordered_nodes])
    metrics = compute_tortuosity_metrics(pts)

    return {
        'seg_ids': seg_ids, 'n_elementary': len(seg_ids), 'length': length,
        'non_simple_path': non_simple_path,
        'endpoints': (ordered_nodes[0], ordered_nodes[-1]), 'metrics': metrics,
    }


def metrics_per_label(G, mask_img, segments, edge_to_seg, node_to_segs):
    """
    Binary mask -> one vessel; multi-class mask -> one vessel per label.
    Returns a list of rows: {label, n_elementary, length, arc_over_chord,
    mean_curvature, mean_square_curvature, rms_curvature, fit_rmse,
    non_simple_path}.
    """
    membership = nodes_in_mask(G, mask_img)
    if isinstance(membership, set):
        membership = {1: membership}

    rows = []
    for label, node_set in membership.items():
        seg_ids = segments_in_vessel(node_set, segments, node_to_segs)
        agg = aggregate_vessel_metrics(seg_ids, segments, G)
        m = agg['metrics']
        rows.append({
            'label': label,
            'n_elementary': agg['n_elementary'],
            'length': agg['length'],
            'arc_over_chord': m.get('arc_over_chord'),
            'mean_curvature': m.get('spline_mean_curvature'),
            'mean_square_curvature': m.get('spline_mean_square_curvature'),
            'rms_curvature': m.get('spline_rms_curvature'),
            'fit_rmse': m.get('fit_rmse'),
            'non_simple_path': agg['non_simple_path'],
        })
    return rows


if __name__ == "__main__":
    # --- Sanity checks on one Eva_Coronaries subject (print only, no CSV) ---
    import os
    import pickle
    import tempfile

    import nibabel as nib

    from Graph_extractor import extract_graph
    from Segment_extractor import build_segments

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
        G = pickle.load(f)['graph']

    segments, edge_to_seg, node_to_segs = build_segments(G)
    total_graph_length = G.size(weight='weight')
    n_non_degenerate = sum(1 for s in segments if not s['degenerate'])

    print(f"\nLoaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"build_segments: {len(segments)} elementary segments "
          f"({len(segments) - n_non_degenerate} degenerate)")

    # --- Trivial binary mask = the whole vessel volume (original nifti thresholded) ---
    orig_nii = nib.load(nifti_path)
    whole_mask_data = (orig_nii.get_fdata() > 0).astype(np.uint8)
    whole_mask_img = nib.Nifti1Image(whole_mask_data, affine=orig_nii.affine, header=orig_nii.header)

    rows = metrics_per_label(G, whole_mask_img, segments, edge_to_seg, node_to_segs)

    print("\n=== Trivial whole-volume binary mask ===")
    print(f"Labels found: {len(rows)}")
    for row in rows:
        coverage = (row['n_elementary'] / n_non_degenerate * 100) if n_non_degenerate else 0.0
        print(f"  label={row['label']}: n_elementary={row['n_elementary']} "
              f"({coverage:.1f}% of {n_non_degenerate} non-degenerate segments)")
        print(f"    length={row['length']:.3f}  (total graph length={total_graph_length:.3f})")
        print(f"    arc_over_chord={row['arc_over_chord']:.4f}, "
              f"mean_curvature={row['mean_curvature']:.4f}, "
              f"mean_square_curvature={row['mean_square_curvature']:.4f}, "
              f"rms_curvature={row['rms_curvature']:.4f}, fit_rmse={row['fit_rmse']:.4f}")
        print(f"    non_simple_path={row['non_simple_path']}")

    assert abs(rows[0]['length'] - total_graph_length) / total_graph_length < 0.02, \
        "Whole-volume vessel length should be ~= total graph length!"
    print("OK: whole-volume vessel length matches the total graph length "
          "(within the small margin from nodes that fall outside the mask's "
          "rounded voxel bounds).")

    assert rows[0]['non_simple_path'], (
        "Expected the whole-volume mask to trigger the non-simple-path "
        "fallback (the full graph is a branching tree, not a single chain)."
    )
    print("OK: whole-volume vessel triggered the non-simple-path WARNING "
          "fallback, as expected for a branching tree (this is the requested "
          "WARNING-triggering case).")

    # --- Synthetic multi-class demo ---
    # Eva_Coronaries only provides binary masks; to also exercise the
    # multi-label loop in metrics_per_label, split the same binary volume
    # into two labels by the median X voxel index. Purely a demo, not a
    # clinically meaningful split.
    print("\n=== Synthetic multi-class demo (split by X-index median) ===")
    nz_idx = np.argwhere(whole_mask_data > 0)
    x_median = np.median(nz_idx[:, 0])
    multi_mask_data = np.zeros_like(whole_mask_data)
    multi_mask_data[nz_idx[:, 0], nz_idx[:, 1], nz_idx[:, 2]] = np.where(
        nz_idx[:, 0] <= x_median, 1, 2
    )
    multi_mask_img = nib.Nifti1Image(multi_mask_data.astype(np.uint8), affine=orig_nii.affine, header=orig_nii.header)

    multi_rows = metrics_per_label(G, multi_mask_img, segments, edge_to_seg, node_to_segs)
    print(f"Labels found: {len(multi_rows)}")
    for row in multi_rows:
        print(f"  label={row['label']}: n_elementary={row['n_elementary']}, "
              f"length={row['length']:.3f}, arc_over_chord={row['arc_over_chord']:.4f}, "
              f"non_simple_path={row['non_simple_path']}")
