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

# Dead-end elementary segments shorter than this (mm) are pruned from a
# reconstructed vessel before checking whether it forms a simple chain (see
# _prune_short_branches / aggregate_vessel_metrics). Rationale: a mask that
# is anatomically a single vessel can still pick up a tiny spurious side
# branch -- a skeletonization artifact, or a hand segmentation that grazes a
# neighbouring vessel at a bifurcation -- which otherwise makes the
# reconstructed subgraph look like a branching tree (non_simple_path=True)
# instead of the clean line it should be. 3.0 mm matches Graph_extractor's
# MERGE_DISTANCE (same order of magnitude already used elsewhere in this
# pipeline to clean up spurious small structures) and was validated against
# CCmaxi_4275_1-Segment_1-label.nii.gz: the one real artifact there was
# 2.89 mm, and nothing between 4-15 mm got pruned (no genuine branch hiding
# in that range) -- not a universally "correct" value, tune per dataset via
# the min_branch_length parameter.
MIN_BRANCH_LENGTH_MM = 3.0


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


def _segment_length(seg, G):
    """Sum of edge weights along one elementary segment's node chain."""
    nodes = seg['nodes']
    return sum(G[u][v]['weight'] for u, v in zip(nodes[:-1], nodes[1:]))


def _prune_short_branches(seg_ids, segments, G, min_branch_length):
    """
    Iteratively drop short dead-end elementary segments from `seg_ids`, so a
    vessel whose induced subgraph has small spurious side-branches (see
    MIN_BRANCH_LENGTH_MM) can collapse back into a clean simple path.

    At each step, among the segments currently touching a degree-1 node of
    the induced subgraph (i.e. a dead end of the reconstructed vessel), the
    SHORTEST one is dropped if its length is below `min_branch_length`.
    Stops as soon as at most 2 dead ends remain (a clean simple path) or as
    soon as the shortest remaining dead-end segment is at or above
    `min_branch_length` (the remaining branching is then treated as real,
    not an artifact, and left for the caller's longest-path fallback).

    Never prunes below 2 segments: a lone segment or a 2-segment chain is
    always left untouched.
    """
    kept = list(seg_ids)

    while len(kept) > 2:
        nodes_union = set()
        for seg_id in kept:
            nodes_union.update(segments[seg_id]['nodes'])
        subG = G.subgraph(nodes_union)
        degree = dict(subG.degree())

        degree_1_nodes = [n for n in subG.nodes() if degree.get(n) == 1]
        if len(degree_1_nodes) <= 2:
            break

        # Exactly one kept segment touches each leaf node, by construction
        # (build_segments partitions G's edges, so only endpoint nodes are
        # shared across segments).
        leaf_seg_ids = {
            seg_id for seg_id in kept
            if degree.get(segments[seg_id]['endpoints'][0]) == 1
            or degree.get(segments[seg_id]['endpoints'][1]) == 1
        }

        shortest_id = min(leaf_seg_ids, key=lambda sid: _segment_length(segments[sid], G))
        shortest_len = _segment_length(segments[shortest_id], G)
        if shortest_len >= min_branch_length:
            break

        logger.info(
            "Pruned short branch: segment %d (%.2f mm < %.2f mm threshold)",
            shortest_id, shortest_len, min_branch_length
        )
        kept.remove(shortest_id)

    return kept


def _aggregate_curvature(length_metrics_pairs):
    """
    Length-weighted aggregation of curvature metrics over an iterable of
    (length, metrics) pairs, where `metrics` is a dict as returned by
    compute_tortuosity_metrics (or None) -- used in place of a global
    full-path spline refit, which fits one spline through the internal
    bifurcation-joint/chunk-boundary nodes and so inflates curvature right
    at those corners (see aggregate_vessel_metrics / aggregate_path_metrics).

    Generic over the source of the pairs: a caller aggregating whole
    elementary segments passes each one's own (_segment_length, seg['metrics']);
    a caller aggregating a path's segment_chain passes each chunk's own
    (entry['length'], entry['metrics']) -- including a partial trailing
    chunk's already-truncated length/metrics, unlike segments[seg_id] which
    always holds the FULL segment.

    Entries with NaN or missing curvature (degenerate, <4 nodes) are
    excluded from both the weighted sum and the weight denominator, so one
    NaN can't poison the whole average. rms_curvature is derived from the
    length-weighted mean of the squared curvature (sqrt of that), not a
    weighted mean of per-entry rms values, to stay mathematically
    consistent with mean_square_curvature.

    Returns (mean_curv, mean_sq_curv, rms_curv, excluded_length_frac).
    If every entry is degenerate (or the iterable is empty), the three
    metrics are NaN and excluded_length_frac is 1.0.
    """
    total_length = 0.0
    excluded_length = 0.0
    weighted_curv = 0.0
    weighted_curv2 = 0.0

    for length, metrics in length_metrics_pairs:
        total_length += length

        curv = metrics.get('spline_mean_curvature') if metrics is not None else None
        curv2 = metrics.get('spline_mean_square_curvature') if metrics is not None else None
        if curv is None or curv2 is None or np.isnan(curv) or np.isnan(curv2):
            excluded_length += length
            continue

        weighted_curv += curv * length
        weighted_curv2 += curv2 * length

    excluded_frac = (excluded_length / total_length) if total_length > 0 else 1.0
    kept_length = total_length - excluded_length
    if kept_length <= 0:
        return np.nan, np.nan, np.nan, excluded_frac

    mean_curv = weighted_curv / kept_length
    mean_curv2 = weighted_curv2 / kept_length
    rms_curv = np.sqrt(mean_curv2)
    return mean_curv, mean_curv2, rms_curv, excluded_frac


def aggregate_vessel_metrics(seg_ids, segments, G, min_branch_length=MIN_BRANCH_LENGTH_MM):
    """
    Reassemble the whole vessel made of the elementary segments `seg_ids` and
    compute its metrics as a single unit. Elementary arc_over_chord values
    are NOT summed (non-additive): the whole-vessel spline is fit once on the
    concatenated, ordered path.

    Curvature (spline_mean_curvature, spline_mean_square_curvature,
    spline_rms_curvature), by contrast, is NOT taken from that full-path
    refit: a single spline through the internal bifurcation-joint nodes
    inflates curvature right at those corners, so for multi-segment vessels
    these three values are overwritten with a length-weighted aggregation of
    the kept segments' own precomputed metrics (see
    _aggregate_curvature). arc_over_chord and the rest of the dict
    (spline_arc_length, spline_chord_length, fit_rmse) still come from the
    full-path refit.

    Before reconstructing the path, short dead-end segments are pruned via
    _prune_short_branches (see MIN_BRANCH_LENGTH_MM): a mask that is
    anatomically a single vessel but picks up a small spurious side-branch
    then still collapses into a clean simple path instead of triggering the
    non-simple-path fallback below.

    Returns a dict: seg_ids (the segments actually used, POST-pruning),
    n_elementary, n_pruned (dead-end segments dropped as short branches),
    length (sum of the kept segments' edge weights), non_simple_path (True
    if the induced subgraph still needed the longest-path fallback after
    pruning, because it wasn't a simple chain), endpoints, metrics (the dict
    returned by compute_tortuosity_metrics, with the curvature fields
    replaced as above and a curv_excluded_length_frac field added for
    multi-segment vessels -- the fraction of vessel length whose segment had
    NaN curvature and was excluded from that aggregation).
    """
    if not seg_ids:
        return {
            'seg_ids': [], 'n_elementary': 0, 'n_pruned': 0, 'length': 0.0,
            'non_simple_path': False, 'endpoints': None,
            'metrics': compute_tortuosity_metrics(np.zeros((0, 3))),
        }

    if len(seg_ids) == 1:
        seg = segments[seg_ids[0]]
        pts = np.array([G.nodes[n]['pos'] for n in seg['nodes']])
        metrics = seg['metrics'] if seg['metrics'] is not None else compute_tortuosity_metrics(pts)
        return {
            'seg_ids': seg_ids, 'n_elementary': 1, 'n_pruned': 0,
            'length': _segment_length(seg, G), 'non_simple_path': False,
            'endpoints': seg['endpoints'], 'metrics': metrics,
        }

    pruned_ids = _prune_short_branches(seg_ids, segments, G, min_branch_length)
    n_pruned = len(seg_ids) - len(pruned_ids)
    length = sum(_segment_length(segments[seg_id], G) for seg_id in pruned_ids)

    nodes_union = set()
    for seg_id in pruned_ids:
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
            "instead of 2 -- it contains internal bifurcations) even after "
            "pruning branches shorter than %.2f mm (%d segment(s) pruned); "
            "falling back to the longest endpoint-to-endpoint path.",
            len(degree_1_nodes), min_branch_length, n_pruned
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

    mean_curv, mean_curv2, rms_curv, excluded_frac = _aggregate_curvature(
        (_segment_length(segments[seg_id], G), segments[seg_id]['metrics'])
        for seg_id in pruned_ids
    )
    metrics['spline_mean_curvature'] = mean_curv
    metrics['spline_mean_square_curvature'] = mean_curv2
    metrics['spline_rms_curvature'] = rms_curv
    metrics['curv_excluded_length_frac'] = excluded_frac

    return {
        'seg_ids': pruned_ids, 'n_elementary': len(pruned_ids), 'n_pruned': n_pruned,
        'length': length, 'non_simple_path': non_simple_path,
        'endpoints': (ordered_nodes[0], ordered_nodes[-1]), 'metrics': metrics,
    }


def aggregate_path_metrics(result):
    """
    One vessel-level row for a single ostium->endpoint path, from a
    Path_analyzer.path_to_point() `result` dict. This is the batch/endpoint
    counterpart of aggregate_vessel_metrics, for when the vessel is defined
    by a query point rather than a mask-derived set of whole segment ids.

    Unlike aggregate_vessel_metrics, there is no subgraph reconstruction, no
    branch pruning and no endpoint search: result['path_nodes'] is already a
    simple ordered chain by construction (a graph shortest path), so
    result['path_metrics'] (the whole-path spline refit) is reused directly
    for arc_over_chord / spline_arc_length / spline_chord_length / fit_rmse
    instead of being recomputed.

    Curvature (mean_curvature, mean_square_curvature, rms_curvature) is, as
    in aggregate_vessel_metrics, NOT taken from that full-path refit but
    overwritten with a length-weighted aggregation (_aggregate_curvature) --
    here fed directly from result['segment_chain'], using each chunk's OWN
    length and metrics. This is the critical difference from simply calling
    aggregate_vessel_metrics(seg_ids, ...) with the chain's seg_ids: the last
    chunk of a path may be 'partial' (the query point snapped to a degree-2
    node inside an elementary segment's interior, see path_to_point /
    Segment_extractor._chunk_metrics), and segment_chain already carries
    that chunk's truncated nodes/length/metrics -- whereas segments[seg_id]
    always holds the segment's FULL node set, which would overshoot past the
    query point if used here.

    Returns a dict: n_elementary, length, arc_over_chord, spline_arc_length,
    spline_chord_length, fit_rmse, mean_curvature, mean_square_curvature,
    rms_curvature, curv_excluded_length_frac, has_partial_tail (True if the
    path's last segment_chain entry is a partial chunk),
    target_snap_distance (mm, from result -- a quick QC signal for how far
    the query point was from any graph node).
    """
    chain = result['segment_chain']
    path_metrics = result['path_metrics']

    length = sum(entry['length'] for entry in chain)
    has_partial_tail = bool(chain) and not chain[-1]['full_segment']

    mean_curv, mean_curv2, rms_curv, excluded_frac = _aggregate_curvature(
        (entry['length'], entry['metrics']) for entry in chain
    )

    return {
        'n_elementary': len(chain),
        'length': length,
        'arc_over_chord': path_metrics.get('arc_over_chord') if path_metrics is not None else None,
        'spline_arc_length': path_metrics.get('spline_arc_length') if path_metrics is not None else None,
        'spline_chord_length': path_metrics.get('spline_chord_length') if path_metrics is not None else None,
        'fit_rmse': path_metrics.get('fit_rmse') if path_metrics is not None else None,
        'mean_curvature': mean_curv,
        'mean_square_curvature': mean_curv2,
        'rms_curvature': rms_curv,
        'curv_excluded_length_frac': excluded_frac,
        'has_partial_tail': has_partial_tail,
        'target_snap_distance': result['target_snap_distance'],
    }


def metrics_per_label(G, mask_img, segments, edge_to_seg, node_to_segs,
                       min_branch_length=MIN_BRANCH_LENGTH_MM):
    """
    Binary mask -> one vessel; multi-class mask -> one vessel per label.
    `min_branch_length` is forwarded to aggregate_vessel_metrics (see
    MIN_BRANCH_LENGTH_MM); pass 0.0 to disable branch pruning entirely.
    Returns a list of rows: {label, n_elementary, n_pruned, length,
    arc_over_chord, mean_curvature, mean_square_curvature, rms_curvature,
    fit_rmse, curv_excluded_length_frac, non_simple_path}.

    `edge_to_seg` is currently unused in this function's body -- reserved
    for a planned Path_analyzer module that will need arbitrary-edge ->
    segment lookups; kept in the signature rather than dropped.
    """
    membership = nodes_in_mask(G, mask_img)
    if isinstance(membership, set):
        membership = {1: membership}

    rows = []
    for label, node_set in membership.items():
        seg_ids = segments_in_vessel(node_set, segments, node_to_segs)
        agg = aggregate_vessel_metrics(seg_ids, segments, G, min_branch_length=min_branch_length)
        m = agg['metrics']
        rows.append({
            'label': label,
            'n_elementary': agg['n_elementary'],
            'n_pruned': agg['n_pruned'],
            'length': agg['length'],
            'arc_over_chord': m.get('arc_over_chord'),
            'mean_curvature': m.get('spline_mean_curvature'),
            'mean_square_curvature': m.get('spline_mean_square_curvature'),
            'rms_curvature': m.get('spline_rms_curvature'),
            'fit_rmse': m.get('fit_rmse'),
            'curv_excluded_length_frac': m.get('curv_excluded_length_frac'),
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

    nifti_path = os.path.join("data_test", "Eva_Coronaries", "CCmaxi_4275.nii")
    scratch_dir = os.path.join(tempfile.gettempdir(), "vessel_aggregator_CCmaxi_4275")
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

    # min_branch_length=0.0: this test's purpose is to verify full-graph
    # coverage/length, so branch pruning (which would drop short dead-end
    # twigs and break the length assertion below) is explicitly disabled
    # here -- unrelated to the pruning demonstrated further down.
    rows = metrics_per_label(G, whole_mask_img, segments, edge_to_seg, node_to_segs, min_branch_length=0.0)

    print("\n=== Trivial whole-volume binary mask (pruning disabled) ===")
    print(f"Labels found: {len(rows)}")
    for row in rows:
        coverage = (row['n_elementary'] / n_non_degenerate * 100) if n_non_degenerate else 0.0
        print(f"  label={row['label']}: n_elementary={row['n_elementary']}, n_pruned={row['n_pruned']} "
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
        print(f"  label={row['label']}: n_elementary={row['n_elementary']}, n_pruned={row['n_pruned']}, "
              f"length={row['length']:.3f}, arc_over_chord={row['arc_over_chord']:.4f}, "
              f"non_simple_path={row['non_simple_path']}")

    # --- Real single-vessel mask (CCmaxi_4275_1-Segment_1-label.nii.gz) ---
    # Unlike the two demos above (whole-volume tree, synthetic median split),
    # this is an actual single-vessel segmentation for the SAME patient as
    # nifti_path, produced independently of the full CCmaxi_4275.nii mask
    # used to build G. nodes_in_mask requires the mask to share G's voxel
    # grid/affine (see module docstring); verified once, offline, that both
    # files are shape (512, 512, 459) with an identical affine -- not
    # re-checked here on every run.
    #
    # Run it twice, with and without branch pruning, to show what
    # MIN_BRANCH_LENGTH_MM actually buys: this real single-vessel mask was
    # previously found to trigger non_simple_path=True (a small spurious
    # side-branch at a bifurcation, not a genuine second vessel) -- pruning
    # should collapse it back into a clean simple path.
    print("\n=== Real single-vessel mask (CCmaxi_4275_1-Segment_1-label.nii.gz) ===")
    real_vessel_mask_path = os.path.join("data_test", "CCmaxi_4275_1-Segment_1-label.nii.gz")
    real_vessel_mask_img = nib.load(real_vessel_mask_path)

    print("-- without branch pruning (min_branch_length=0.0) --")
    real_rows_unpruned = metrics_per_label(
        G, real_vessel_mask_img, segments, edge_to_seg, node_to_segs, min_branch_length=0.0
    )
    for row in real_rows_unpruned:
        print(f"  label={row['label']}: n_elementary={row['n_elementary']}, n_pruned={row['n_pruned']}, "
              f"length={row['length']:.3f} mm, non_simple_path={row['non_simple_path']}")

    print(f"-- with branch pruning (min_branch_length={MIN_BRANCH_LENGTH_MM} mm, the default) --")
    real_rows = metrics_per_label(G, real_vessel_mask_img, segments, edge_to_seg, node_to_segs)
    print(f"Labels found: {len(real_rows)}")
    for row in real_rows:
        print(f"  label={row['label']}: n_elementary={row['n_elementary']}, n_pruned={row['n_pruned']}, "
              f"length={row['length']:.3f} mm")
        print(f"    arc_over_chord={row['arc_over_chord']:.4f}, "
              f"mean_curvature={row['mean_curvature']:.4f}, "
              f"mean_square_curvature={row['mean_square_curvature']:.4f}, "
              f"rms_curvature={row['rms_curvature']:.4f}, fit_rmse={row['fit_rmse']:.4f}")
        print(f"    non_simple_path={row['non_simple_path']}")
        if row['non_simple_path']:
            print("    NOTE: still not a simple chain even after pruning branches < "
                  f"{MIN_BRANCH_LENGTH_MM} mm ({row['n_pruned']} segment(s) pruned) -- the "
                  "remaining branching is likely real (or MIN_BRANCH_LENGTH_MM is too low "
                  "for this mask), not just a skeletonization/segmentation artifact.")
        else:
            print(f"    OK: single simple chain found after pruning {row['n_pruned']} "
                  "short branch segment(s), as expected for a genuine single-vessel mask.")
