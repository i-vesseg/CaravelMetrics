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


def _terminal_direction(pts_window):
    """
    Least-squares dominant direction (unit vector) of an ordered point
    window: the first principal axis (SVD) of the centered points, oriented
    along the window's actual traversal (first point -> last point). Used
    instead of a single node-to-node step vector so voxel-level jitter in
    the underlying skeleton doesn't dominate the tangent estimate right at
    a junction.
    """
    pts_window = np.asarray(pts_window)
    if len(pts_window) < 2:
        return None
    centered = pts_window - pts_window.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    if np.dot(direction, pts_window[-1] - pts_window[0]) < 0:
        direction = -direction
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return None
    return direction / norm


def _discrete_polyline_turning(pts):
    """
    Fallback total turning (radians) for a segment too short to fit a cubic
    spline (compute_tortuosity_metrics returns spline_mean_curvature=NaN for
    <4 points, see Metric_extractor.py): sum of the discrete angles between
    consecutive edge vectors along the segment's own node chain, so such
    segments still contribute to total_turning instead of being silently
    dropped.
    """
    pts = np.asarray(pts)
    if len(pts) < 3:
        return 0.0
    vecs = np.diff(pts, axis=0)
    norms = np.linalg.norm(vecs, axis=1)
    total = 0.0
    for k in range(len(vecs) - 1):
        n1, n2 = norms[k], norms[k + 1]
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_angle = np.dot(vecs[k], vecs[k + 1]) / (n1 * n2)
        total += np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return float(total)


def _segment_turning(seg_metrics, seg_pts):
    """
    A single elementary segment's own turning (radians): its precomputed
    spline_mean_curvature (see _path_turning docstring -- despite the name,
    it is already an INTEGRAL of curvature over the segment's arc length,
    i.e. a turning angle in radians, not a per-length mean) if finite, else
    the discrete polyline fallback (_discrete_polyline_turning) computed
    from the segment's own node positions -- covers degenerate segments
    (<4 nodes, where seg_metrics is None or spline_mean_curvature is NaN) so
    they still contribute to total_turning.
    """
    turning = seg_metrics.get('spline_mean_curvature') if seg_metrics is not None else None
    if turning is None or np.isnan(turning):
        return _discrete_polyline_turning(seg_pts)
    return float(turning)


def _path_turning(ordered_pts, junction_indices, seg_turnings, window=3):
    """
    Junction-aware total turning of a reconstructed vessel/path, recovering
    the turn angle concentrated exactly AT a bifurcation/junction node --
    invisible to _aggregate_curvature's length-weighted mean because a
    junction node is only ever an ENDPOINT shared between two elementary
    segments, never an interior sample of either one's own spline fit (see
    the module's segment partition: every node of degree>=3 is by
    construction a segment boundary, not an interior node). Two straight
    segments meeting at a sharp angle at their shared junction would
    otherwise aggregate to ~0 mean curvature despite the vessel visibly
    turning there.

        total_turning = sum(seg_turnings) + junction_turning

    where seg_turnings are each segment's own turning (see _segment_turning)
    and junction_turning sums, over each junction index i in
    `junction_indices` (positions into `ordered_pts`), the angle between the
    least-squares incoming tangent direction (_terminal_direction over
    ordered_pts[max(0, i-window):i+1]) and the outgoing tangent direction
    (_terminal_direction over ordered_pts[i:i+window+1]). This is a discrete
    tangent DISCONTINUITY at the joint that is not part of either adjacent
    segment's own integral, so adding it here is not double-counting the
    same turning twice.

    Returns (total_turning, junction_turning, n_junctions) -- n_junctions is
    the number of junction indices for which both terminal directions could
    actually be estimated (excludes degenerate near-zero-length windows).
    """
    junction_angles = []
    for i in junction_indices:
        incoming = _terminal_direction(ordered_pts[max(0, i - window):i + 1])
        outgoing = _terminal_direction(ordered_pts[i:i + window + 1])
        if incoming is None or outgoing is None:
            continue
        cos_angle = np.dot(incoming, outgoing)
        junction_angles.append(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    junction_turning = float(np.sum(junction_angles)) if junction_angles else 0.0
    n_junctions = len(junction_angles)
    seg_turning_total = float(np.sum(seg_turnings)) if len(seg_turnings) else 0.0
    total_turning = seg_turning_total + junction_turning
    return total_turning, junction_turning, n_junctions


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

    metrics also always carries 4 junction-aware turning fields (see
    _path_turning): total_turning and junction_turning (radians),
    n_junctions, and soam = total_turning / length (rad/mm, NaN if
    length <= 0) -- these recover the turn angle concentrated exactly at an
    internal bifurcation/junction node of the reconstructed vessel, which
    the length-weighted curvature aggregation above cannot see (a junction
    node is only ever a segment ENDPOINT, never an interior spline sample).
    """
    if not seg_ids:
        empty_metrics = compute_tortuosity_metrics(np.zeros((0, 3)))
        empty_metrics['total_turning'] = np.nan
        empty_metrics['junction_turning'] = np.nan
        empty_metrics['n_junctions'] = 0
        empty_metrics['soam'] = np.nan
        return {
            'seg_ids': [], 'n_elementary': 0, 'n_pruned': 0, 'length': 0.0,
            'non_simple_path': False, 'endpoints': None,
            'metrics': empty_metrics,
        }

    if len(seg_ids) == 1:
        seg = segments[seg_ids[0]]
        pts = np.array([G.nodes[n]['pos'] for n in seg['nodes']])
        # Copy rather than reuse seg['metrics'] directly: it's the SAME dict
        # object stored on `segments`, shared across every vessel that
        # includes this segment -- mutating it in place below (adding the 4
        # turning fields) would leak vessel-level fields into that shared,
        # per-segment cache.
        metrics = (
            dict(seg['metrics']) if seg['metrics'] is not None
            else compute_tortuosity_metrics(pts)
        )
        length = _segment_length(seg, G)
        turning = _segment_turning(seg['metrics'], pts)
        metrics['total_turning'] = turning
        metrics['junction_turning'] = 0.0
        metrics['n_junctions'] = 0
        metrics['soam'] = (turning / length) if length > 0 else np.nan
        return {
            'seg_ids': seg_ids, 'n_elementary': 1, 'n_pruned': 0,
            'length': length, 'non_simple_path': False,
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

    # Junction-aware turning (see _path_turning): the pruned segments'
    # own turnings, plus the turn angle at each internal junction node of
    # `ordered_nodes` -- i.e. every degree>=3 node strictly between the two
    # path endpoints (degree is looked up on the FULL graph G, not subG,
    # since a junction's other branches were pruned out of subG but still
    # make it a real anatomical bifurcation).
    junction_indices = [
        i for i in range(1, len(ordered_nodes) - 1)
        if G.degree(ordered_nodes[i]) >= 3
    ]
    seg_turnings = [
        _segment_turning(
            segments[seg_id]['metrics'],
            np.array([G.nodes[n]['pos'] for n in segments[seg_id]['nodes']]),
        )
        for seg_id in pruned_ids
    ]
    total_turning, junction_turning, n_junctions = _path_turning(pts, junction_indices, seg_turnings)
    metrics['total_turning'] = total_turning
    metrics['junction_turning'] = junction_turning
    metrics['n_junctions'] = n_junctions
    metrics['soam'] = (total_turning / length) if length > 0 else np.nan

    return {
        'seg_ids': pruned_ids, 'n_elementary': len(pruned_ids), 'n_pruned': n_pruned,
        'length': length, 'non_simple_path': non_simple_path,
        'endpoints': (ordered_nodes[0], ordered_nodes[-1]), 'metrics': metrics,
    }


def aggregate_path_metrics(result, G=None):
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

    Parameters:
        result : dict, from Path_analyzer.path_to_point.
        G : networkx.Graph, optional. If given, also computes the 4
            junction-aware turning fields below, using result['path_nodes']
            (degree looked up on G) and each segment_chain entry's own
            nodes/metrics -- same method as aggregate_vessel_metrics, see
            _path_turning. If None (default), those 4 fields are set to
            None, so existing callers that don't pass G keep working
            unchanged.

    Returns a dict: n_elementary, length, arc_over_chord, spline_arc_length,
    spline_chord_length, fit_rmse, mean_curvature, mean_square_curvature,
    rms_curvature, curv_excluded_length_frac, has_partial_tail (True if the
    path's last segment_chain entry is a partial chunk),
    target_snap_distance (mm, from result -- a quick QC signal for how far
    the query point was from any graph node), total_turning, junction_turning,
    n_junctions, soam (radians / rad-per-mm, see _path_turning; None if G is
    not given).
    """
    chain = result['segment_chain']
    path_metrics = result['path_metrics']

    length = sum(entry['length'] for entry in chain)
    has_partial_tail = bool(chain) and not chain[-1]['full_segment']

    mean_curv, mean_curv2, rms_curv, excluded_frac = _aggregate_curvature(
        (entry['length'], entry['metrics']) for entry in chain
    )

    if G is None:
        total_turning = junction_turning = n_junctions = soam = None
    else:
        path_nodes = result['path_nodes']
        path_pts = np.array([G.nodes[n]['pos'] for n in path_nodes])
        junction_indices = [
            i for i in range(1, len(path_nodes) - 1)
            if G.degree(path_nodes[i]) >= 3
        ]
        seg_turnings = [
            _segment_turning(
                entry['metrics'],
                np.array([G.nodes[n]['pos'] for n in entry['nodes']]),
            )
            for entry in chain
        ]
        total_turning, junction_turning, n_junctions = _path_turning(
            path_pts, junction_indices, seg_turnings
        )
        soam = (total_turning / length) if length > 0 else np.nan

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
        'total_turning': total_turning,
        'junction_turning': junction_turning,
        'n_junctions': n_junctions,
        'soam': soam,
    }


def metrics_per_label(G, mask_img, segments, edge_to_seg, node_to_segs,
                       min_branch_length=MIN_BRANCH_LENGTH_MM):
    """
    Binary mask -> one vessel; multi-class mask -> one vessel per label.
    `min_branch_length` is forwarded to aggregate_vessel_metrics (see
    MIN_BRANCH_LENGTH_MM); pass 0.0 to disable branch pruning entirely.
    Returns a list of rows: {label, n_elementary, n_pruned, length,
    arc_over_chord, mean_curvature, mean_square_curvature, rms_curvature,
    fit_rmse, curv_excluded_length_frac, non_simple_path, total_turning,
    junction_turning, n_junctions, soam}. The last 4 are the junction-aware
    turning fields from aggregate_vessel_metrics (see _path_turning).

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
            'total_turning': m.get('total_turning'),
            'junction_turning': m.get('junction_turning'),
            'n_junctions': m.get('n_junctions'),
            'soam': m.get('soam'),
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

    # --- Junction-aware turning (total_turning / junction_turning / n_junctions / soam) ---
    print("\n=== Junction-aware turning fields ===")

    main_row = real_rows[0]
    print(f"Real single-vessel mask (pruned, main vessel): "
          f"total_turning={main_row['total_turning']:.4f} rad, "
          f"junction_turning={main_row['junction_turning']:.4f} rad, "
          f"n_junctions={main_row['n_junctions']}, soam={main_row['soam']:.6f} rad/mm "
          f"(mean_curvature={main_row['mean_curvature']:.4f} for comparison)")

    assert main_row['junction_turning'] >= 0, "junction_turning must be non-negative."
    assert np.isfinite(main_row['total_turning']) and np.isfinite(main_row['soam']), \
        "total_turning/soam should be finite (not NaN) for the main, non-degenerate vessel."
    assert main_row['total_turning'] >= main_row['junction_turning'], (
        "total_turning (segment turnings + junction_turning) can never be smaller "
        "than junction_turning alone, since per-segment turnings are non-negative."
    )
    print("OK: junction_turning >= 0 and total_turning/soam are finite for the main vessel, "
          "with total_turning >= junction_turning as expected.")

    # --- Synthetic sharp-bend-at-a-bifurcation demo ---
    # Real Eva_Coronaries masks don't reliably contain a "two straight
    # segments meeting at a known sharp angle at a junction" case on demand,
    # so this builds one directly: two perfectly straight 10 mm segments
    # (A along +x, B along +y) sharing a junction node J, plus a short (1.5
    # mm) third stub C off J so J is a genuine degree-3 bifurcation in G.
    # This is exactly the blind spot _path_turning is meant to fix: A and B
    # each have ~0 curvature on their own, so the pre-existing length-weighted
    # mean_curvature aggregation reports ~0 for the whole vessel despite its
    # visible 90-degree turn at J -- total_turning/soam should not.
    print("\n=== Synthetic sharp-bend-at-a-bifurcation demo (isolates the fix) ===")

    def _make_synth_segment(seg_id, nodes, G_synth):
        pts = np.array([G_synth.nodes[n]['pos'] for n in nodes])
        degenerate = len(nodes) < 4
        seg_metrics = None if degenerate else compute_tortuosity_metrics(pts)
        return {'id': seg_id, 'nodes': nodes, 'endpoints': (nodes[0], nodes[-1]),
                'degenerate': degenerate, 'metrics': seg_metrics}

    def _add_chain(G_synth, ids, positions):
        for nid, pos in zip(ids, positions):
            if nid not in G_synth:
                G_synth.add_node(nid, pos=np.array(pos, dtype=float))
        for u, v in zip(ids[:-1], ids[1:]):
            d = np.linalg.norm(G_synth.nodes[u]['pos'] - G_synth.nodes[v]['pos'])
            G_synth.add_edge(u, v, weight=float(d))

    G_synth = nx.Graph()
    a_ids = [f'a{i}' for i in range(11)]
    _add_chain(G_synth, a_ids, [(float(i), 0.0, 0.0) for i in range(11)])
    junction = a_ids[-1]  # (10, 0, 0)

    b_ids = [junction] + [f'b{i}' for i in range(1, 11)]
    _add_chain(G_synth, b_ids, [(10.0, float(i), 0.0) for i in range(11)])

    c_ids = [junction] + [f'c{i}' for i in range(1, 4)]
    _add_chain(G_synth, c_ids, [(10.0, 0.0, i * 0.5) for i in range(4)])

    segments_synth = [
        _make_synth_segment(0, a_ids, G_synth),
        _make_synth_segment(1, b_ids, G_synth),
        _make_synth_segment(2, c_ids, G_synth),
    ]

    synth_agg = aggregate_vessel_metrics([0, 1, 2], segments_synth, G_synth)
    synth_m = synth_agg['metrics']
    print(f"Two straight 10 mm segments at a 90-degree junction (stub C pruned as a short branch):")
    print(f"  non_simple_path={synth_agg['non_simple_path']}, n_pruned={synth_agg['n_pruned']}")
    print(f"  mean_curvature={synth_m['spline_mean_curvature']:.6f} (length-weighted, should be ~0)")
    print(f"  total_turning={synth_m['total_turning']:.4f} rad, "
          f"junction_turning={synth_m['junction_turning']:.4f} rad (expected ~pi/2={np.pi/2:.4f}), "
          f"n_junctions={synth_m['n_junctions']}, soam={synth_m['soam']:.6f} rad/mm")

    assert not synth_agg['non_simple_path'], "Stub C should have been pruned, leaving a clean A-B chain."
    assert abs(synth_m['spline_mean_curvature']) < 1e-6, \
        "The two straight segments' own aggregated mean_curvature should be ~0."
    assert synth_m['n_junctions'] == 1, "Exactly one internal junction (J) is expected on the A-B path."
    assert abs(synth_m['junction_turning'] - np.pi / 2) < 1e-3, \
        "junction_turning should recover the ~90-degree bend at J."
    assert synth_m['soam'] > 0.01, (
        "soam must clearly be > 0 for this vessel even though mean_curvature is ~0 -- "
        "this is the exact junction-blindness case total_turning/soam is meant to fix."
    )
    print("OK: mean_curvature is ~0 for this vessel, but soam/junction_turning correctly "
          "recover the 90-degree bend at the bifurcation -- the exact case this fixes.")

    single_agg = aggregate_vessel_metrics([0], segments_synth, G_synth)
    single_m = single_agg['metrics']
    print(f"\nSingle-segment vessel (segment A alone): junction_turning={single_m['junction_turning']}, "
          f"n_junctions={single_m['n_junctions']}, total_turning={single_m['total_turning']:.6f}")
    assert single_m['junction_turning'] == 0.0 and single_m['n_junctions'] == 0, \
        "A single-segment vessel has no internal junction, so junction_turning must be exactly 0.0."
    print("OK: single-segment vessels report junction_turning == 0.0, n_junctions == 0.")

    empty_agg = aggregate_vessel_metrics([], segments_synth, G_synth)
    empty_m = empty_agg['metrics']
    print(f"\nEmpty vessel (no segments): total_turning={empty_m['total_turning']}, "
          f"junction_turning={empty_m['junction_turning']}, soam={empty_m['soam']}")
    assert np.isnan(empty_m['total_turning']) and np.isnan(empty_m['junction_turning']) \
        and np.isnan(empty_m['soam']) and empty_m['n_junctions'] == 0, \
        "An empty vessel should report NaN turning/soam and n_junctions == 0."
    print("OK: empty vessel reports NaN total_turning/junction_turning/soam and n_junctions == 0.")
