# -*- coding: utf-8 -*-
"""
Metric_extractor.py

Analyze a saved 3D vessel graph: decompose each connected component into
anatomical branch segments and compute the requested morphometric metrics
(length, volume, bifurcations, fractal/lacunarity, per-branch tortuosity).
"""
import argparse
import os
import networkx as nx
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
import pickle
import vedo
from pathlib import Path

import Branch_metrics as BM


def load_label_map(txt_path):
    labels = {}
    if txt_path is None or not os.path.exists(txt_path):
        print("  ! Warning: No valid label map provided. Using default names.")
        return labels
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    # The file is tab-separated. Column 1 is ID, Column 2 is the Acronym.
    for line in lines[1:]:  # Skip the header row
        parts = line.strip().split('\t')
        if len(parts) >= 3 and parts[1].isdigit():
            labels[int(parts[1])] = parts[2].strip()
            
    return labels

_CURV_KEYS = ('spline_mean_curvature', 'spline_mean_square_curvature', 'spline_rms_curvature')
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')


def compute_tortuosity_metrics(points, sigma=0.25, end_trim=2.0, min_length=6.0,
                               n_samples=500, smoothing=None):
    """Mean curvature of a branch on a *smoothing* spline through its nodes.

    Curvature is undefined on a raw polyline, so a spline is needed; but an
    interpolating spline (s=0) follows the voxel staircase of the skeleton and
    returns ~0.1-0.3 /mm of noise whatever the true curvature. The spline here
    may deviate from the nodes by an RMS of `sigma` mm (s = n * sigma^2).

    end_trim   arc length (mm) excluded at each end: the skeleton bends into
               bifurcation centres and erodes at tips.
    min_length segments shorter than this (spline arc length, mm) give NaN.
    smoothing  explicit splprep `s`; overrides sigma when given.

    Returns true means (integral over the trimmed arc / its length). Its arc
    length is only the integration measure; reported lengths use the
    geodesic polyline length. Validated on phantoms.
    """
    nan = {k: np.nan for k in _CURV_KEYS}
    pts = np.asarray(points, float)
    # Consecutive duplicate points crash splprep; drop them (keep the first).
    if pts.shape[0] > 1:
        pts = pts[np.insert(np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6, 0, True)]
    if pts.shape[0] < 4:
        return nan
    s = len(pts) * sigma ** 2 if smoothing is None else smoothing
    try:
        tck, _ = splprep(pts.T, s=s)
    except Exception:
        return nan

    u = np.linspace(0, 1, 4 * n_samples)
    speed = np.linalg.norm(np.array(splev(u, tck, der=1)).T, axis=1)
    s_cum = np.r_[0, np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(u))]
    arc_len = s_cum[-1]
    if arc_len < max(min_length, 2 * end_trim + 1e-6):
        return nan

    s_eval = np.linspace(end_trim, arc_len - end_trim, n_samples)
    u_eval = np.interp(s_eval, s_cum, u)
    d1 = np.array(splev(u_eval, tck, der=1)).T
    d2 = np.array(splev(u_eval, tck, der=2)).T
    curvature = (np.linalg.norm(np.cross(d1, d2), axis=1)
                 / np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-12))
    L = s_eval[-1] - s_eval[0]
    mean_curv = _trapz(curvature, s_eval) / L
    mean_sq_curv = _trapz(curvature ** 2, s_eval) / L
    return {
        'spline_mean_curvature': float(mean_curv),
        'spline_mean_square_curvature': float(mean_sq_curv),
        'spline_rms_curvature': float(np.sqrt(mean_sq_curv)),
    }


def _centreline_points(G, step=0.1):
    """Points every `step` mm along all edges of G (the centreline as a
    curve set, independent of how densely the graph nodes are spaced)."""
    if G.number_of_edges() == 0:
        return np.empty((0, 3))
    # every node once, plus the interior points of each edge (no point is
    # repeated, so the count per mm does not depend on node spacing)
    seg = [np.array([G.nodes[n]['pos'] for n in G.nodes() if G.degree(n) > 0], float)]
    for u, v in G.edges():
        p, q = np.asarray(G.nodes[u]['pos'], float), np.asarray(G.nodes[v]['pos'], float)
        m = int(np.ceil(np.linalg.norm(q - p) / step))
        if m > 1:
            seg.append(p + np.outer(np.arange(1, m) / m, q - p))
    return np.vstack(seg)


def compute_fractal_dimension(G, min_box=1.0, max_box=16.0, n_sizes=9, step=0.1, return_r2=False):
    """Box-counting dimension of the centreline over a FIXED range of box
    sizes in mm, so values are comparable between components and subjects.

    The previous version scaled the box sizes to each component's extent and
    counted nodes only, which made FD track component size (a straight 10 mm
    vessel scored 0.72). The range is shrunk to extent/4 for small graphs, and
    NaN is returned when fewer than 2 octaves remain. Counts are averaged over
    three grid offsets.
    """
    pts = _centreline_points(G, step)
    nan = (np.nan, np.nan) if return_r2 else np.nan
    if len(pts) == 0:
        return nan
    lo, hi = max(min_box, 4 * step), min(max_box, np.ptp(pts, axis=0).max() / 4)
    if hi < 4 * lo:
        return nan
    sizes = np.logspace(np.log10(lo), np.log10(hi), n_sizes)
    p0 = pts - pts.min(axis=0)
    counts = [np.mean([len(np.unique(np.floor((p0 + o * sz) / sz).astype(np.int64), axis=0))
                       for o in (0.0, 0.33, 0.67)]) for sz in sizes]
    x, y = np.log(sizes), np.log(counts)
    slope, icpt = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (slope * x + icpt)) ** 2) / np.sum((y - y.mean()) ** 2)
    return (float(-slope), float(r2)) if return_r2 else float(-slope)


def calculate_lacunarity(G, box_size=5.0, step=0.1):
    """Fixed-grid lacunarity, Var/Mean^2 + 1 of the centreline *length* per
    box, with a fixed box size in mm. Length (not node count) is used so the
    value does not depend on node spacing; the box size is fixed (not 1/10 of
    each component's extent) so values are comparable between graphs.
    NaN when the graph spans fewer than 2 boxes."""
    pts = _centreline_points(G, step)
    if len(pts) == 0:
        return np.nan
    shifted = pts - pts.min(axis=0)
    num_boxes = np.floor(shifted.max(axis=0) / box_size).astype(int) + 1
    if np.prod(num_boxes) < 2:
        return np.nan
    flat = np.ravel_multi_index(np.floor(shifted / box_size).astype(int).T, num_boxes)
    counts = np.bincount(flat, minlength=int(np.prod(num_boxes))).astype(float)
    mean_val = counts.mean()
    return float(counts.var() / mean_val ** 2 + 1) if mean_val > 0 else np.nan

def analyze_component_structure(G_comp):
    """Loops (cyclomatic number), and degree >= 4 nodes split by origin:
    'merged' ones were created by Graph_extractor's ladder-rung contraction
    (two junctions < 1 mm apart merged into one, node attribute 'merged'),
    the others came out of skeletonisation as is.
    Only the latter are reported as abnormal; previously every merge was
    counted as an anomaly (ADAM test mask: 3 -> 18)."""
    num_loops = len(nx.cycle_basis(G_comp))
    high = [n for n, d in G_comp.degree() if d > 3]
    # 'merged' (current) or networkx's 'contraction' (graphs from earlier versions)
    merged = [n for n in high if G_comp.nodes[n].get('merged') or 'contraction' in G_comp.nodes[n]]
    return num_loops, len(high) - len(merged), len(merged)

def extract_branch_segments(G_comp):
    """
    Decompose a component into anatomical branch segments and return their
    node chains (each an ordered list of node ids).

    A segment is the maximal centerline chain between two consecutive
    topological nodes -- bifurcations (degree >= 3) or endpoints (degree 1);
    every interior node has degree 2. This standard VMTK/literature "branch"
    decomposition is unique and root-independent, so each vessel segment is
    emitted exactly once with no shared-trunk overlap.
    """
    special = {n for n, deg in G_comp.degree() if deg != 2}
    segments = []
    visited_edges = set()

    def _edge(a, b):
        return (a, b) if a <= b else (b, a)

    # Ordinary branches: from each special node, walk out along every incident
    # edge through degree-2 conduits until the next special node.
    for s in special:
        for nb in G_comp.neighbors(s):
            if _edge(s, nb) in visited_edges:
                continue
            chain = [s, nb]
            visited_edges.add(_edge(s, nb))
            prev, cur = s, nb
            while cur not in special:
                nxt = next(n for n in G_comp.neighbors(cur) if n != prev)
                visited_edges.add(_edge(cur, nxt))
                chain.append(nxt)
                prev, cur = cur, nxt
            segments.append(chain)

    # Pure loops: edges still unvisited form cycles with no special node (an
    # isolated ring); emit each ring once as a closed chain.
    remaining = {_edge(u, v) for u, v in G_comp.edges()} - visited_edges
    while remaining:
        u0, v0 = next(iter(remaining))
        chain = [u0, v0]
        visited_edges.add(_edge(u0, v0))
        remaining.discard(_edge(u0, v0))
        cur = v0
        while cur != u0:
            nbrs = [n for n in G_comp.neighbors(cur) if _edge(cur, n) not in visited_edges]
            if not nbrs:
                break
            nxt = nbrs[0]
            visited_edges.add(_edge(cur, nxt))
            remaining.discard(_edge(cur, nxt))
            chain.append(nxt)
            cur = nxt
        segments.append(chain)

    return segments


def network_metrics(G):
    """Metrics defined on a whole (possibly disconnected) network rather
    than per component: fractal dimension (fixed 1-16 mm scale range, with
    its fit R^2) and lacunarity (fixed 5 mm boxes)."""
    fd, r2 = compute_fractal_dimension(G, return_r2=True)
    return {'fractal_dimension': fd, 'fractal_dimension_r2': r2, 'lacunarity': calculate_lacunarity(G)}


COMPONENT_KEYS = [
    'total_length', 'num_bifurcations', 'bifurcation_density', 'volume',
    'fractal_dimension', 'lacunarity', 'num_loops', 'num_abnormal_degree_nodes', 'num_merged_junctions',
    'avg_diameter', 'tortuosity_index', 'num_segments',
    'num_bifurcations_measured', 'num_bifurcations_short_loop', 'median_bifurcation_angle',
]
SEGMENT_COLUMNS = [
    'geodesic_length', 'chord_length', 'tortuosity_index', 'avg_diameter',
    'spline_mean_curvature', 'spline_mean_square_curvature', 'spline_rms_curvature',
]


def _component_row(data):
    row = {k: data.get(k, np.nan) for k in COMPONENT_KEYS}
    agg = data.get('aggregated_tortuosity', {})
    row['mean_curvature'] = agg.get('mean_curvature', np.nan)
    row['mean_square_curvature'] = agg.get('mean_square_curvature', np.nan)
    return row


def summarize_components(components, network=None, true_volume=None):
    """One summary row from a list of component dicts (a region, or the whole
    graph): counts and lengths summed, curvature / diameter length-weighted
    over components, tortuosity index and junction values pooled over all
    segments and junctions, network-level FD/lacunarity, and the mask volume
    if given."""
    comps = sorted(components, key=lambda d: d.get('total_length', 0), reverse=True)
    row = {'num_components': len(comps)}
    for k in ('total_length', 'num_bifurcations', 'volume', 'num_loops', 'num_abnormal_degree_nodes',
              'num_merged_junctions', 'num_segments'):
        row[k] = float(np.nansum([d.get(k, np.nan) for d in comps])) if comps else 0.0
    tl = row['total_length']
    row['bifurcation_density'] = row['num_bifurcations'] / tl if tl > 0 else np.nan
    lw = {'mean_curvature': lambda d: d.get('aggregated_tortuosity', {}).get('mean_curvature', np.nan),
          'mean_square_curvature': lambda d: d.get('aggregated_tortuosity', {}).get('mean_square_curvature', np.nan),
          'avg_diameter': lambda d: d.get('avg_diameter', np.nan)}
    for k, get in lw.items():
        v = np.array([get(d) for d in comps], float)
        # curvature: weight by the length on which it is defined
        wkey = 'curvature_length' if 'curvature' in k else 'total_length'
        w = np.array([d.get(wkey, d.get('total_length', 0)) for d in comps], float)
        ok = np.isfinite(v) & (w > 0)
        row[k] = float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else np.nan
    row.update(pooled_summary(comps))
    row.update(network or {})
    if true_volume is not None:
        row['volume_approx_cylinder'] = row['volume']
        row['volume'] = true_volume
    return row


def save_results(results, output_folder, region_true_volumes=None, patient_name=None,
                 region_network_metrics=None):
    """Write the metric tables:

      <patient>_all_components_by_region.csv   one row per connected component
      <patient>_region_summary.csv             one row per region ('global' without atlas)
      <patient>_segments.csv                   one row per branch segment
      <patient>_bifurcations.csv               one row per measured junction

    region_network_metrics: {region_label: network_metrics(region graph)},
    written to the region summary (FD/lacunarity are computed on the region's
    whole network, not averaged over components).
    """
    print("  Saving results...")
    file_prefix = f"{patient_name}_" if patient_name else ""
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    region_rows, seg_rows, bif_rows = [], [], []
    region_results = results.get('region_metrics', {'global': results})
    if 'global' in region_results: print("  No regional metrics found, saving global results only.")

    region_components = {}
    for region_label, region_data in region_results.items():
        if not isinstance(region_data, dict): continue
        components = {k: v for k, v in region_data.items() if isinstance(v, dict) and 'total_length' in v}
        region_components[region_label] = list(components.values())
        sorted_items = sorted(components.items(), key=lambda item: item[1].get('total_length', 0), reverse=True)
        for comp_idx, (_, data) in enumerate(sorted_items, start=1):
            row = {'region_label': region_label, 'component_index': comp_idx}
            row.update(_component_row(data))
            region_rows.append(row)
            for seg_idx, e in enumerate(data.get('segments', []), start=1):
                sm = e.get('segment_metrics', {})
                seg_rows.append({'region_label': region_label, 'component_index': comp_idx, 'segment_index': seg_idx,
                                 **{k: sm.get(k, np.nan) for k in SEGMENT_COLUMNS}})
            for b in data.get('bifurcations', []):
                bif_rows.append({'region_label': region_label, 'component_index': comp_idx,
                                 'x': b['pos'][0], 'y': b['pos'][1], 'z': b['pos'][2],
                                 **{k: b[k] for k in ('bifurcation_angle', 'deflection_1', 'deflection_2', 'on_short_loop')}})

    pd.DataFrame(seg_rows, columns=['region_label', 'component_index', 'segment_index'] + SEGMENT_COLUMNS).to_csv(
        os.path.join(output_folder, f'{file_prefix}segments.csv'), index=False)
    pd.DataFrame(bif_rows, columns=['region_label', 'component_index', 'x', 'y', 'z',
                                    'bifurcation_angle', 'deflection_1', 'deflection_2', 'on_short_loop']).to_csv(
        os.path.join(output_folder, f'{file_prefix}bifurcations.csv'), index=False)

    if region_rows:
        df_detailed = pd.DataFrame(region_rows)
        df_detailed = df_detailed.sort_values(by=['region_label', 'total_length'], ascending=[True, False])
        df_detailed.to_csv(
            os.path.join(output_folder, f'{file_prefix}all_components_by_region.csv'), index=False)

        summary_rows = []
        for region_label in sorted(region_components, key=str):
            tv = (region_true_volumes or {}).get(region_label)
            row = {'region_label': region_label}
            row.update(summarize_components(region_components[region_label],
                                            (region_network_metrics or {}).get(region_label), tv))
            summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows).sort_values(by='region_label')
        df_summary.to_csv(os.path.join(output_folder, f'{file_prefix}region_summary.csv'), index=False)

    print(f"Results saved to: {output_folder}")


def ensure_edge_weights(G):
    """
    Guarantee every edge carries a physical 'weight' (length in mm).

    networkx silently substitutes 1 for a missing weight, which would
    quietly turn total_length/volume/geodesic metrics into edge counts with
    no warning. Graph_extractor always writes 'weight', but graphs loaded
    from other sources (robust_graph_loader scenarios 1/2/4) may not.
    Recomputes missing weights from node positions; raises if a graph has
    neither weights nor positions to recompute them from.
    """
    missing = [(u, v) for u, v, d in G.edges(data=True) if 'weight' not in d]
    if not missing:
        return 0
    if not all('pos' in G.nodes[n] for e in missing for n in e):
        raise ValueError(
            f"{len(missing)} edges lack a 'weight' attribute and their nodes "
            "have no 'pos' to recompute it from; length/volume/geodesic "
            "metrics cannot be computed for this graph."
        )
    for u, v in missing:
        G[u][v]['weight'] = float(
            np.linalg.norm(np.asarray(G.nodes[u]['pos']) - np.asarray(G.nodes[v]['pos']))
        )
    print(f"  ! {len(missing)} edges had no 'weight'; recomputed from node positions.")
    return len(missing)


TORTUOSITY_MIN_LENGTH_MM = 6.0


def _bifurcation_summary(bifs):
    """Median bifurcation angle over junctions (pooled). Junctions whose
    daughters rejoin within a short loop (mask tunnels, tiny anastomoses;
    Branch_metrics.SHORT_LOOP_MM) are counted but not in the median."""
    ang = [b['bifurcation_angle'] for b in bifs if not b.get('on_short_loop', False)]
    return {'num_bifurcations_measured': len(ang),
            'num_bifurcations_short_loop': len(bifs) - len(ang),
            'median_bifurcation_angle': float(np.median(ang)) if ang else np.nan}


def pooled_summary(component_dicts):
    """Region/global-level values pooled over the components' segments and
    junctions (not averages of component medians)."""
    bifs = [b for d in component_dicts for b in d.get('bifurcations', [])]
    segs = [e.get('segment_metrics', {}) for d in component_dicts for e in d.get('segments', [])]
    out = _bifurcation_summary(bifs)
    out['tortuosity_index'] = median_tortuosity_index(component_dicts)
    L = np.array([m.get('geodesic_length', 0.0) for m in segs], float)
    ok = np.array([np.isfinite(m.get('spline_mean_curvature', np.nan)) for m in segs], bool)
    out['curvature_length_coverage'] = float(L[ok].sum() / L.sum()) if L.sum() > 0 else np.nan
    return out


def median_tortuosity_index(component_dicts):
    """Median arc/chord over all segments >= TORTUOSITY_MIN_LENGTH_MM of the
    given components (segments pooled, not averaged per component).

    A median, not a length-weighted mean: a few near-closed segments (chord
    -> 0, e.g. a vessel that leaves and re-enters the same bifurcation) have
    arc/chord in the tens to hundreds and would dominate a mean (TRE test
    graph: mean 2.18 vs median 1.27)."""
    vals = [sm['tortuosity_index'] for d in component_dicts for e in d.get('segments', [])
            for sm in (e.get('segment_metrics', {}),)
            if sm.get('geodesic_length', 0) >= TORTUOSITY_MIN_LENGTH_MM and np.isfinite(sm.get('tortuosity_index', np.inf))]
    return float(np.median(vals)) if vals else np.nan


def compute_metrics_for_mask(G, node_radius_map):
    """Compute the full metric set for every connected component of G. All
    metrics are always computed -- there is no selection knob."""
    results = {}
    ensure_edge_weights(G)
    for cid, comp_nodes in enumerate(nx.connected_components(G)):
        Gc = G.subgraph(comp_nodes).copy()
        data = {}

        total_len = Gc.size(weight='weight')
        degrees = dict(Gc.degree())
        num_bif = sum(1 for d in degrees.values() if d >= 3)

        data['total_length'] = total_len
        data['num_bifurcations'] = num_bif
        data['bifurcation_density'] = num_bif / total_len if total_len > 0 else np.nan

        # Volume: conical frustum per edge (not a mean-radius cylinder, which
        # under-estimates volume wherever the radius tapers).
        vol = 0.0
        for u, v, d in Gc.edges(data=True):
            l = d.get('weight', 0)
            r_u = node_radius_map.get(u, 0.5)
            r_v = node_radius_map.get(v, 0.5)
            vol += np.pi * l * (r_u ** 2 + r_u * r_v + r_v ** 2) / 3.0
        data['volume'] = vol

        data['fractal_dimension'] = compute_fractal_dimension(Gc)
        data['lacunarity'] = calculate_lacunarity(Gc)

        nl, nab, nmerged = analyze_component_structure(Gc)
        data['num_loops'] = nl
        data['num_abnormal_degree_nodes'] = nab
        data['num_merged_junctions'] = nmerged

        # Unique branch decomposition: each segment appears once (no trunk
        # overlap), so no traversal-count down-weighting is needed.
        seg_list = extract_branch_segments(Gc)
        seg_metrics = []

        # Length-weighted curvature accumulators. A <4-point segment gives
        # NaN curvature, so it must drop from BOTH numerator and denominator:
        # each metric sums only the length of segments that defined it.
        len_curv = len_curv2 = 0.0
        sum_curv = sum_curv2 = 0.0

        edt_rmap = {n: float(Gc.nodes[n].get('radius_edt', node_radius_map.get(n, np.nan))) for n in Gc.nodes()}

        def _segment_tortuosity(chain):
            pts = np.array([Gc.nodes[n]['pos'] for n in chain])
            seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            L_geo = float(sum(seg_lens))  # sequential add: bit-stable geodesic length
            sm = {'geodesic_length': L_geo}
            # Tortuosity index: geodesic path length over the straight-line
            # chord, using the same geodesic length as above. Valid even for
            # <4-node segments (unlike spline curvature, which needs >=4).
            chord = float(np.linalg.norm(pts[-1] - pts[0])) if len(pts) > 1 else 0.0
            sm['chord_length'] = chord
            sm['tortuosity_index'] = (L_geo / chord) if chord > 0 else np.inf
            # Length-weighted mean diameter (each node weighted by half its
            # incident edge lengths, so uneven spacing doesn't bias).
            diam = np.array([2 * node_radius_map.get(n, 0.5) for n in chain])
            w = np.zeros(len(chain))
            w[:-1] += seg_lens / 2.0
            w[1:] += seg_lens / 2.0
            sm['avg_diameter'] = float(np.average(diam, weights=w)) if w.sum() > 0 else float(diam.mean())
            tort = compute_tortuosity_metrics(pts)
            sm.update(tort)  # spline_mean/mean_square/rms_curvature (NaN < 6 mm)
            return sm, L_geo, tort

        # Length-weighted mean of per-segment diameter.
        len_diam = sum_diam = 0.0

        for seg in seg_list:
            sm, L_geo, tort = _segment_tortuosity(seg)
            if L_geo > 0 and np.isfinite(sm['avg_diameter']):
                sum_diam += sm['avg_diameter'] * L_geo
                len_diam += L_geo
            if not np.isnan(tort['spline_mean_curvature']):
                sum_curv += tort['spline_mean_curvature'] * L_geo
                len_curv += L_geo
            if not np.isnan(tort['spline_mean_square_curvature']):
                sum_curv2 += tort['spline_mean_square_curvature'] * L_geo
                len_curv2 += L_geo
            seg_metrics.append(sm)

        data['segments'] = [{'segment_metrics': sm} for sm in seg_metrics]
        data['num_segments'] = len(seg_metrics)
        data['avg_diameter'] = (sum_diam / len_diam) if len_diam > 0 else np.nan
        data['tortuosity_index'] = median_tortuosity_index([data])
        data['curvature_length'] = len_curv
        data['aggregated_tortuosity'] = {
            'mean_curvature': (sum_curv / len_curv) if len_curv > 0 else np.nan,
            'mean_square_curvature': (sum_curv2 / len_curv2) if len_curv2 > 0 else np.nan,
        }

        # Bifurcations: branching angles per junction.
        bifs = BM.bifurcation_metrics(Gc, node_radius_map, seg_list, junction_radius=edt_rmap)
        data['bifurcations'] = bifs
        data.update(_bifurcation_summary(bifs))

        results[cid] = data
    return results


def _parse_tre_tubes(path):
    """
    Parse the point sequence of every 'Tube' object in a MetaIO/TubeTK .tre
    scene file. Returns a list of (N, 4) arrays of [x, y, z, radius] per
    tube, in file order. Only Tube point blocks are read; everything else
    in the MetaIO format is ignored.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    tubes = []
    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip() == 'ObjectType = Tube':
            npoints, j = None, i + 1
            while j < n and not lines[j].strip().startswith('Points ='):
                s = lines[j].strip()
                if s.startswith('NPoints'):
                    npoints = int(s.split('=')[1].strip())
                j += 1
            if npoints is None:
                raise ValueError(f"{path}: Tube object with no NPoints (line {i + 1}).")
            j += 1  # skip the 'Points =' line itself
            pts = np.array([[float(v) for v in lines[j + k].split()[:4]] for k in range(npoints)])
            tubes.append(pts)
            i = j + npoints
        else:
            i += 1
    return tubes


def _load_tre_scene(path, stitch_tolerance_mm=2.0):
    """
    Build a Caravel-style graph from a MetaIO/TubeTK .tre scene: chains each
    tube's points into a path, then stitches tubes together wherever an
    endpoint lands within stitch_tolerance_mm of an already-placed point --
    .tre ParentID only encodes group membership, not branch-to-branch
    topology, so touching tubes have to be found geometrically. Tubes that
    don't touch anything within tolerance stay as separate components, like
    any other pipeline input with gaps in the underlying data.
    """
    tubes = _parse_tre_tubes(path)
    G = nx.Graph()
    node_radius_map = {}
    all_pos = np.empty((0, 3))
    node_id = 0

    for tube in tubes:
        tube_node_ids = list(range(node_id, node_id + len(tube)))
        for (x, y, z, r), nid in zip(tube, tube_node_ids):
            pos = np.array([x, y, z])
            G.add_node(nid, pos=pos, radius=r)
            node_radius_map[nid] = r
        G.add_edges_from(zip(tube_node_ids[:-1], tube_node_ids[1:]))
        node_id += len(tube)

        tube_xyz = tube[:, :3]
        if all_pos.shape[0] > 0:
            tree = cKDTree(all_pos)
            for end_idx in (0, -1):
                dist, nearest = tree.query(tube_xyz[end_idx])
                if dist < stitch_tolerance_mm:
                    G.add_edge(tube_node_ids[end_idx], int(nearest))
        all_pos = np.vstack([all_pos, tube_xyz])

    for u, v in G.edges():
        G[u][v]['weight'] = float(np.linalg.norm(G.nodes[u]['pos'] - G.nodes[v]['pos']))

    return G, node_radius_map


def robust_graph_loader(pkl_path):
    """
    Intelligently loads a graph from a pickle file, hunting for networkx objects.
    Includes support for Caravel dictionaries, raw NetworkX graphs, UNC Skwerer
    Tree dictionaries, and MetaIO/TubeTK .tre scenes.
    """
    if str(pkl_path).lower().endswith('.tre'):
        return _load_tre_scene(pkl_path)

    # Legacy-pickle compatibility: map numpy._core (NumPy >=2.0) back to
    # numpy.core so arrays pickled under older NumPy still unpickle.
    import sys
    if 'numpy._core' not in sys.modules:
        try:
            import numpy.core as _core
            sys.modules['numpy._core'] = _core
        except: pass

    with open(pkl_path, 'rb') as f:
        try: 
            data = pickle.load(f, encoding='latin1')
        except:
            f.seek(0)
            data = pickle.load(f)
            
    G = None
    node_radius_map = None

    # Scenario 1: The user saved a raw networkx graph directly
    if isinstance(data, (nx.Graph, nx.DiGraph)):
        G = data
        
    # Scenario 2: The user saved a dictionary (like the Caravel pipeline)
    elif isinstance(data, dict) and 'graph' in data and isinstance(data['graph'], (nx.Graph, nx.DiGraph)):
        G = data['graph']
        if 'node_radius_map' in data:
            node_radius_map = data['node_radius_map']

    # Scenario 3: UNC Skwerer/Bullitt Tree Dictionary
    elif isinstance(data, dict) and 'nodes' in data and isinstance(data['nodes'], list):
        G = nx.Graph()
        node_radius_map = {}
        
        for node in data['nodes']:
            n_id = node['id']
            pos = np.array(node['xyz'])
            radius = node['radius']
            
            G.add_node(n_id, pos=pos, radius=radius)
            node_radius_map[n_id] = radius
            
            parent = node.get('parent', -1)
            if parent >= 0:
                G.add_edge(n_id, parent)
                
        # Calculate edge weights (Euclidean distance) required for metric extraction
        for u, v in G.edges():
            G[u][v]['weight'] = np.linalg.norm(G.nodes[u]['pos'] - G.nodes[v]['pos'])

    # Scenario 4: Hunt for any value that is a networkx graph
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (nx.Graph, nx.DiGraph)):
                G = value
                break

    if G is None:
        raise ValueError(f"Could not find a valid networkx Graph in {pkl_path}")
    if G.is_directed():
        # Every metric here is orientation-free, and connected_components()
        # raises on directed graphs.
        G = G.to_undirected()

    # Fallback: Build radius map from node attributes if not provided separately
    if node_radius_map is None:
        node_radius_map = {}
        for n in G.nodes():
            node_radius_map[n] = G.nodes[n].get('radius', 0.5) 

    return G, node_radius_map

def region_graphs(G, node_label):
    """Split G into one graph per atlas region: the subgraph induced by each
    region's nodes (node_label: {node: region id}, 0 = outside every region).
    Edges between two regions are dropped, so a vessel crossing a border is
    cut there. Returns {region id: graph}."""
    by = {}
    for n, r in node_label.items():
        if r > 0:
            by.setdefault(r, []).append(n)
    return {r: G.subgraph(ns).copy() for r, ns in by.items()}


# QC flags: (column, test, threshold, meaning). Thresholds set from clean vs
# noisy phantoms and the test masks.
QC_FLAGS = [
    ('uncovered_volume_fraction', '>', 0.05, 'centreline misses >5 % of the mask volume'),
    ('graph_to_mask_volume', 'outside', (0.85, 1.15), 'graph volume differs from mask volume by >15 %'),
    ('small_component_length_fraction', '>', 0.10, '>10 % of length in fragments < 10 mm'),
    ('short_terminal_branches_per_100mm', '>', 1.0, 'many short terminal branches (spurs: noisy mask?)'),
    ('curvature_length_coverage', '<', 0.60, 'curvature defined on <60 % of the length'),
    ('fractal_dimension_r2', '<', 0.98, 'box-counting fit poor (FD unreliable)'),
    ('facing_endpoint_pairs', '>', 0, 'wide vessel ends facing each other across a gap (vessel interrupted?)'),
]


def qc_row(graph_package, summary):
    """Flat QC record: the graph-extraction QC stored in the pkl (if the graph
    came from Graph_extractor) plus metric-level checks, and the list of
    raised flags."""
    row = dict((graph_package or {}).get('qc', {}))
    tl = summary.get('total_length', np.nan)
    if 'short_terminal_branches' in row and tl and tl > 0:
        row['short_terminal_branches_per_100mm'] = 100.0 * row['short_terminal_branches'] / tl
    for k in ('total_length', 'num_components', 'curvature_length_coverage', 'fractal_dimension_r2',
              'num_bifurcations_measured', 'num_bifurcations_short_loop'):
        if k in summary:
            row.setdefault(k, summary[k])
    flags = []
    for col, op, thr, meaning in QC_FLAGS:
        v = row.get(col, np.nan)
        if v is None or not np.isfinite(v):
            continue
        bad = (v > thr) if op == '>' else (v < thr) if op == '<' else (v < thr[0] or v > thr[1])
        if bad:
            flags.append(f"{col}: {meaning}")
    row['qc_flags'] = '; '.join(flags)
    row['qc_pass'] = not flags
    return row


def _default(fn, name):
    import inspect
    return inspect.signature(fn).parameters[name].default


def _write_run_info(output_folder, patient_name, graph_pkl_path, graph_package, extra):
    import json
    import Graph_extractor as _GE
    info = {'graph_file': os.path.abspath(graph_pkl_path),
            'graph_code_version': (graph_package or {}).get('code_version', 'unknown'),
            'graph_params': (graph_package or {}).get('params', {}),
            'metric_code_version': _GE.code_version(),
            'metric_settings': {'curvature_sigma_mm': _default(compute_tortuosity_metrics, 'sigma'),
                                'curvature_end_trim_mm': _default(compute_tortuosity_metrics, 'end_trim'),
                                'curvature_min_length_mm': _default(compute_tortuosity_metrics, 'min_length'),
                                'fd_box_range_mm': [_default(compute_fractal_dimension, 'min_box'),
                                                    _default(compute_fractal_dimension, 'max_box')],
                                'lacunarity_box_mm': _default(calculate_lacunarity, 'box_size'),
                                'tortuosity_min_length_mm': TORTUOSITY_MIN_LENGTH_MM,
                                'bifurcation_end_exclusion_mm': BM.END_EXCLUSION_MM,
                                'bifurcation_window_mm': BM.BIF_WINDOW_MM}}
    info.update(extra)
    with open(os.path.join(output_folder, f'{patient_name}_run_info.json'), 'w') as f:
        json.dump(info, f, indent=1, default=str)


def _load_package(path):
    try:
        with open(path, 'rb') as f:
            d = pickle.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def extract_metrics(output_folder, graph_pkl_path, label_map_path=None, mask_path=None, registered_atlas_path=None):
    use_atlas = True if registered_atlas_path else False
    results = {'region_metrics': {}} if use_atlas else {}
    
    # Auto-extract patient name from the graph filename
    patient_name = os.path.basename(graph_pkl_path)
    if patient_name.endswith('_vessel_data.pkl'):
        patient_name = patient_name[: -len('_vessel_data.pkl')]
    else:
        patient_name = os.path.splitext(patient_name)[0]
    
    try:
        G, node_radius_map = robust_graph_loader(graph_pkl_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load graph from {graph_pkl_path}: {e}") from e
    package = _load_package(graph_pkl_path) if str(graph_pkl_path).endswith('.pkl') else {}
    
    # Optional true mask: gives an exact voxel-count volume instead of the
    # cylinder/frustum graph approximation.
    true_mask_data = None
    voxel_volume_mm3 = 0.0
    total_true_volume = np.nan

    if mask_path and os.path.exists(mask_path):
        try:
            mask_img = nib.load(mask_path)
            true_mask_data = np.asarray(mask_img.dataobj) > 0
            voxel_volume_mm3 = np.prod(mask_img.header.get_zooms()[:3])
            total_true_volume = np.sum(true_mask_data) * voxel_volume_mm3
            print(f"  Loaded True Mask for exact volume computation (Voxel size: {voxel_volume_mm3:.4f} mm³)")
        except Exception as e:
            print(f"  Warning: Could not load true mask: {e}")
    
    label_map = load_label_map(label_map_path)

    if use_atlas:
        # Load Atlas with Nibabel instead of SimpleITK to avoid LPS/RAS coordinate flipping!
        aligned_atlas_img = nib.load(registered_atlas_path)
        aligned_atlas_data = np.round(aligned_atlas_img.get_fdata(dtype=np.float32)).astype(np.int16)
        inv_affine = np.linalg.inv(aligned_atlas_img.affine)
        
        node_label = {}
        size = aligned_atlas_data.shape
        
        for n in G.nodes():
            pos = G.nodes[n]['pos']
            try:
                # Convert RAS physical position to Voxel Index directly
                idx_float = nib.affines.apply_affine(inv_affine, pos)
                idx = np.round(idx_float).astype(int)
                
                # Check bounds
                if (0 <= idx[0] < size[0] and 0 <= idx[1] < size[1] and 0 <= idx[2] < size[2]):
                    region_id = aligned_atlas_data[idx[0], idx[1], idx[2]]

                    # Any positive label is a region -- atlas-agnostic, works
                    # for any parcellation. 0 is the standard background/
                    # unlabeled voxel value.
                    node_label[n] = int(region_id)
            except Exception as e:
                raise ValueError(f"Failed to transform point {pos} to index in atlas. Error: {e}")
            
        ensure_edge_weights(G)
        regions = region_graphs(G, node_label)
        region_true_volumes = {}
        region_net = {}
        for r_id in sorted(regions.keys()):
            r_id_int = int(r_id) # Absolute safety cast for dictionary matching

            sub_G = regions[r_id]
            region_result = compute_metrics_for_mask(sub_G, node_radius_map)
            region_net[r_id_int] = network_metrics(sub_G)

            # Record True Regional Volume (applied once at region-summary level
            # in save_results, not per-component -- see region_true_volumes).
            if true_mask_data is not None:
                region_voxel_count = np.sum(true_mask_data & (aligned_atlas_data == r_id))
                region_true_volumes[r_id_int] = region_voxel_count * voxel_volume_mm3

            results['region_metrics'][r_id_int] = region_result

        save_results(results, output_folder, region_true_volumes=region_true_volumes, patient_name=patient_name,
                     region_network_metrics=region_net)

        n_regions = len(regions)
        print(f"  Found {n_regions} arterial regions. Generating regional VTPs for {patient_name}...")
        regional_actors = []
        regions_vtp_dir = os.path.join(output_folder, "regional_vtps")
        os.makedirs(regions_vtp_dir, exist_ok=True)
        
        for r_id, sub_G in regions.items():
            r_id_int = int(r_id)
            s_pts = [sub_G.nodes[u]['pos'] for u, v in sub_G.edges()]
            e_pts = [sub_G.nodes[v]['pos'] for u, v in sub_G.edges()]
            
            if s_pts:
                region_lines = vedo.Lines(s_pts, e_pts).lw(3)
                n_cells = region_lines.dataset.GetNumberOfCells()
                visual_id = (r_id_int * 137) % 256
                region_array = np.full(n_cells, visual_id, dtype=np.int32)
                region_lines.celldata["RegionID"] = region_array
                regional_actors.append(region_lines)
                
                # Fetch exact name for file export
                region_name = label_map.get(r_id_int, f"region_{r_id_int}")
                individual_vtp_path = os.path.join(regions_vtp_dir, f"region_{r_id_int}_{region_name}.vtp")
                region_lines.dataset.GetCellData().SetActiveScalars("RegionID")
                region_lines.write(individual_vtp_path)

        if regional_actors:
            combined_vtp = vedo.merge(regional_actors)
            regional_vtp_path = os.path.join(output_folder, f"{patient_name}_regional.vtp")
            combined_vtp.dataset.GetCellData().SetActiveScalars("RegionID")
            combined_vtp.write(regional_vtp_path)
            print(f"  -> Saved combined VTP with {n_regions} regions to: {regional_vtp_path}")
            
        # Global totals over the full unparcellated graph (parcellation splits
        # vessels across regions, so global != sum of regions).
        print("  Computing true global metrics on the full unparcellated graph...")
        true_global_result = compute_metrics_for_mask(G, node_radius_map)

        global_data = summarize_components(list(true_global_result.values()), network_metrics(G),
                                           total_true_volume if true_mask_data is not None else None)

        global_df = pd.DataFrame([global_data])
        global_df.to_csv(os.path.join(output_folder, f'{patient_name}_global_summary.csv'), index=False)
        summary_for_qc = global_data
    
    else:
        print("  Computing global vessel metrics...")
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        global_result = compute_metrics_for_mask(G, node_radius_map)

        # Record True Global Volume under the 'global' pseudo-region key that
        # save_results() uses when there's no atlas (applied once at the
        # region-summary level, not per-component).
        region_true_volumes = {}
        if true_mask_data is not None:
            total_voxel_count = np.sum(true_mask_data)
            region_true_volumes['global'] = total_voxel_count * voxel_volume_mm3

        results.update(global_result)
        net = network_metrics(G)
        save_results(results, output_folder, region_true_volumes=region_true_volumes, patient_name=patient_name,
                     region_network_metrics={'global': net})
        summary_for_qc = summarize_components(list(global_result.values()), net)

    pd.DataFrame([qc_row(package, summary_for_qc)]).to_csv(
        os.path.join(output_folder, f'{patient_name}_qc.csv'), index=False)
    _write_run_info(output_folder, patient_name, graph_pkl_path, package,
                    {'mask': mask_path, 'atlas': registered_atlas_path})

# =========================================================================
#                         STANDALONE CLI EXECUTION
# =========================================================================
if __name__ == "__main__":
    # Dynamically locate the default label map
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_LABEL_MAP = os.path.join(SCRIPT_DIR, "Atlas", "ArterialAtlasLables.txt")

    parser = argparse.ArgumentParser(description="Extract Metrics from Saved Vessel Graph")
    
    parser.add_argument("--graph", required=True, help="Path to the .pkl graph file.")
    parser.add_argument("--out_dir", required=True, help="Directory to save the CSV metrics files.")
    parser.add_argument("--mask", default=None, help="Path to original binary mask for EXACT volume calculation (optional).")
    parser.add_argument("--label_map", default=DEFAULT_LABEL_MAP, help="Path to ArterialAtlasLables.txt (optional).")
    parser.add_argument("--atlas", default=None, help="Path to the registered atlas NIfTI (optional).")

    args = parser.parse_args()

    extract_metrics(
        output_folder=args.out_dir,
        graph_pkl_path=args.graph,
        label_map_path=args.label_map,
        mask_path=args.mask,
        registered_atlas_path=args.atlas
    )