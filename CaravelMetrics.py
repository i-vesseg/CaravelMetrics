# -*- coding: utf-8 -*-
"""
vessel_analysis_modular.py

A command-line tool to analyze 3D vessel masks by skeletonizing,
building a graph, extracting segments, and computing only requested metrics.

Supports two analysis modes:
1. Whole-brain analysis: Analyzes entire vessel network as single region
2. Atlas-based regional analysis: Parcellates vessels into anatomical territories

USAGE:
    # Whole-brain mode:
    python vessel_analysis_modular.py <image_path> <mask_path> [OPTIONS]
    
    # Atlas-based regional mode:
    python vessel_analysis_modular.py <image_path> <mask_path> --image_T1 <T1_path> --atlas_path <atlas_path> [OPTIONS]

ARGUMENTS:
    image_path       Path to vessel image (.nii/.nii.gz)
    mask_path        Path to vessel mask (.nii/.nii.gz)
    --image_T1       (optional) Path to T1 image for atlas registration
    --atlas_path     (optional) Path to atlas file for regional parcellation
    --metrics        (optional) List of metrics to compute
    --output_folder  (optional) Path to save results
    --no_segment_masks      Disable segment mask construction and saving
    --no_conn_comp_masks    Disable connected component reconstruction saving
"""
import os
import argparse
import nibabel as nib
import numpy as np
import pandas as pd
import networkx as nx
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import splprep, splev, interp1d
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from nipype.interfaces import fsl



def compute_tortuosity_metrics(points, smoothing=0, n_samples=500, counts=None):
    """
    Compute tortuosity metrics for a 3D curve defined by 'points' using a cubic B-spline.
    This function reparameterizes the spline by arc length to ensure a uniform-speed curve,
    and can optionally down-weight curvature by per-point occurrence counts.

    Parameters:
      points : array-like, shape (N, 3)
        Input list of 3D curve points.
      smoothing : float, optional
        Smoothing factor for spline fitting (default: 0).
      n_samples : int, optional
        Number of samples along the curve for evaluation (default: 500).
      counts : array-like, shape (N,), optional
        Occurrence counts n(x) at each original input point.  After
        interpolation this yields n(s) at each sampled s, and we will
        weight curvature as κ(s)/n(s).

    Returns:
      dict of tortuosity metrics:
          - spline_arc_length
          - spline_chord_length
          - spline_mean_curvature       (weighted)
          - spline_mean_square_curvature (weighted)
          - spline_rms_curvature        (weighted)
          - arc_over_chord
          - fit_rmse
    """
    pts = np.asarray(points)
    if pts.shape[0] < 4:
        # Not enough points to fit a cubic B-spline
        nan_dict = {k: np.nan for k in [
            'spline_arc_length','spline_chord_length',
            'spline_mean_curvature','spline_mean_square_curvature',
            'spline_rms_curvature','arc_over_chord','fit_rmse']}
        return nan_dict

    # 1) Fit spline and get original u-parameters
    tck, u = splprep(pts.T, s=smoothing)

    # 2) Dense evaluation to compute arc length
    u_fine   = np.linspace(0, 1, n_samples)
    deriv1   = np.array(splev(u_fine, tck, der=1)).T
    du       = np.gradient(u_fine)
    ds       = np.linalg.norm(deriv1, axis=1) * du
    s_cum    = np.cumsum(ds) - ds[0]
    arc_len  = s_cum[-1]

    # 3) Reparameterize by arc length → uniform s samples
    u_of_s   = interp1d(s_cum, u_fine, kind='linear',
                        bounds_error=False, fill_value=(0,1))
    s_uniform= np.linspace(0, arc_len, n_samples)
    u_uniform= u_of_s(s_uniform)
    pts_u    = np.array(splev(u_uniform, tck)).T

    # 4) Compute derivatives wrt s
    dt  = s_uniform[1] - s_uniform[0]
    d1  = np.gradient(pts_u, dt, axis=0)
    d2  = np.gradient(d1, dt, axis=0)

    # 5) Build the weight function n(s) by interpolating original counts if given
    if counts is not None:
        counts_orig  = np.asarray(counts)
        interp_cnt   = interp1d(u, counts_orig, kind='linear',
                                bounds_error=False,
                                fill_value=(counts_orig[0], counts_orig[-1]))
        n_s          = interp_cnt(u_uniform)
        n_s          = np.where(n_s <= 0, 1, n_s)  # clamp to ≥1
    else:
        n_s = np.ones(n_samples)

    # 6) Compute the standard curvature κ(s)
    cross_vec = np.cross(d1, d2)
    speed     = np.linalg.norm(d1, axis=1)
    # add epsilon to avoid div‐by‐zero in speed**3
    curvature = np.linalg.norm(cross_vec, axis=1) / (speed**3 + 1e-10)

    # 7) Form the weighted curvature and its square
    curv_w    = curvature / n_s
    curv2_w   = (curvature**2) / (n_s**2)

    # 8) Integrate weighted curvature over s
    mean_curv       = np.trapz(curv_w,  s_uniform)
    mean_sq_curv    = np.trapz(curv2_w, s_uniform)
    rms_curv        = np.sqrt(mean_sq_curv / arc_len) if arc_len>0 else 0

    # 9) Chord length & fit RMSE
    chord_len = np.linalg.norm(pts_u[-1] - pts_u[0])
    spline_at_u = np.array(splev(u, tck)).T
    fit_rmse    = np.sqrt(np.mean(np.sum((spline_at_u - pts)**2, axis=1)))

    return {
        'spline_arc_length':           arc_len,
        'spline_chord_length':         chord_len,
        'spline_mean_curvature':       mean_curv,
        'spline_mean_square_curvature':mean_sq_curv,
        'spline_rms_curvature':        rms_curv,
        'arc_over_chord':              (arc_len / chord_len) if chord_len>0 else np.inf,
        'fit_rmse':                    fit_rmse
    }


def fractal_dimension(points, box_sizes=None):
    """
    Compute fractal dimension using the box counting method for a set of 3D points.
    'points' is an array of shape (N, 3). Returns the fractal dimension.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return np.nan

    # Shift points to positive coordinates
    mins = points.min(axis=0)
    shifted = points - mins
    maxs = shifted.max(axis=0)
    max_dim = max(maxs)

    # Guard against zero‐extent point clouds:
    if max_dim == 0:
        return np.nan

    # Define box sizes logarithmically if not provided
    if box_sizes is None:
        # Use 10 sizes from a fraction of max_dim to max_dim
        box_sizes = np.logspace(
            np.log10(max_dim / 50.0), 
            np.log10(max_dim), 
            num=10,
            base=10.0
        )
    
    counts = []
    for size in box_sizes:
        if size <= 0 or np.isnan(size):
            # Skip invalid sizes
            continue
        # Determine the number of boxes in each dimension
        bins = np.ceil(maxs / size).astype(int) + 1
        # Compute box indices
        indices = np.floor(shifted / size).astype(int)
        # Unique boxes that contain at least one point
        unique_boxes = {tuple(idx) for idx in indices}
        counts.append(len(unique_boxes))
    
    # If we couldn't collect any valid counts, bail out
    if len(counts) == 0:
        return np.nan

    # Fit a line to the log-log plot of (1/box_size) vs counts
    X = np.log(1.0 / np.array(box_sizes[:len(counts)])).reshape(-1, 1)
    y = np.log(counts)
    reg = LinearRegression().fit(X, y)
    return reg.coef_[0]


def calculate_lacunarity(points, box_size):
    """
    Estimate lacunarity for a set of 3D points using a grid of a given box_size.
    Here we build a grid covering the points and calculate the mean and variance
    of the count of points per box.
    """
    points = np.asarray(points)
    mins = points.min(axis=0)
    shifted = points - mins
    maxs = shifted.max(axis=0)
    # Determine number of boxes per dimension
    num_boxes = np.ceil(maxs / box_size).astype(int) + 1
    grid = np.zeros(num_boxes)
    
    # For each point, increment its corresponding box
    indices = np.floor(shifted / box_size).astype(int)
    for idx in indices:
        grid[tuple(idx)] += 1
    counts = grid.flatten()
    mean_val = counts.mean()
    var_val = counts.var()
    # A common lacunarity measure:
    lac = var_val / (mean_val**2) + 1 if mean_val != 0 else np.nan
    return lac


def analyze_component_structure(G_comp):
    """
    Calculates the number of loops and the number of nodes with abnormal degree (> 3)
    in a connected component.

    Parameters:
        G_comp: networkx.Graph
            A subgraph representing a connected component.

    Returns:
        tuple: (num_loops, num_abnormal_degree_nodes)
    """
    num_loops = len(nx.cycle_basis(G_comp))
    abnormal_degree_nodes = [node for node, degree in G_comp.degree() if degree > 3]
    num_abnormal_degree_nodes = len(abnormal_degree_nodes)
    return num_loops, num_abnormal_degree_nodes


def extract_segments_from_component_using_shortest_path(G_comp, distance_map):
    """
    From the connected component G_comp, identify all endpoints (degree == 1)
    and select three roots:
      1. Endpoint with the largest diameter
      2. Endpoint with the second-largest diameter
      3. Bifurcation node (degree >= 3) with the largest diameter

    Then, compute shortest-path segments from each root to all other endpoints,
    and for each root, return both:
      - A list of shortest-path segments (each a list of nodes) from the root
        to every other endpoint.
      - A mapping from each node to the number of times it appears across those segments.

    Parameters:
      G_comp : networkx.Graph
          A subgraph representing a connected component.
      distance_map : dict-like
          Mapping from node (as tuple) to its radius value (in same units as graph coords).

    Returns:
      segment_lists : List[List[List[node]]]
          A list of three lists, each containing shortest-path segments
          (node lists) from one selected root to every other endpoint.
      segment_counts : List[Dict[node, int]]
          A list of three dictionaries, each corresponding to one selected root.
          Each dictionary maps nodes to their frequency of occurrence in all
          shortest-path segments from that root to every other endpoint.
    """
    # Identify endpoints (degree == 1)
    endpoints = [node for node, deg in G_comp.degree() if deg == 1]
    if not endpoints:
        return [[], [], []], [{}, {}, {}]

    # Diameter at a node = 2 * radius
    diameters_eps = {node: 2 * distance_map[tuple(node)] for node in endpoints}
    # Sort endpoints by diameter descending
    sorted_eps = sorted(endpoints, key=lambda n: diameters_eps[n], reverse=True)

    # First two roots: largest and second-largest diameter endpoints
    root1 = sorted_eps[0]
    root2 = sorted_eps[1] if len(sorted_eps) > 1 else root1

    # Third root: bifurcation (degree >= 3) with largest diameter, or fallback
    bif_nodes = [node for node, deg in G_comp.degree() if deg >= 3]
    if bif_nodes:
        diameters_bif = {node: 2 * distance_map[tuple(node)] for node in bif_nodes}
        root3 = max(bif_nodes, key=lambda n: diameters_bif[n])
    else:
        root3 = root1

    roots = [root1, root2, root3]
    segment_lists = []
    segment_counts_list = []

    # For each selected root, compute segments and counts
    for root in roots:
        segs = []
        counts = defaultdict(int)
        for ep in endpoints:
            if ep == root:
                continue
            try:
                path = nx.shortest_path(G_comp, source=root, target=ep)
            except nx.NetworkXNoPath:
                continue
            segs.append(path)
            # Count occurrences of each node along the path
            for node in path:
                counts[node] += 1
        segment_lists.append(segs)
        segment_counts_list.append(dict(counts))

    return segment_lists, segment_counts_list


def build_graph(skeleton):
    """Build a graph from a skeleton with 26-connectivity."""
    G = nx.Graph()
    shape = skeleton.shape
    fibers = np.argwhere(skeleton)
    for v in fibers:
        coord = tuple(v)
        G.add_node(coord)
        x, y, z = coord
        for i in range(max(0, x-1), min(shape[0], x+2)):
            for j in range(max(0, y-1), min(shape[1], y+2)):
                for k in range(max(0, z-1), min(shape[2], z+2)):
                    if (i, j, k) != (x, y, z) and skeleton[i, j, k]:
                        G.add_edge(coord, (i, j, k), weight=np.linalg.norm(np.array(coord) - np.array((i, j, k))))
    return G


def prune_graph(G):
    """
    Prune G by detecting all simple cycles of length 3 (triangles) and removing the
    heaviest edge in each triangle (based on 'weight').
    Modifies G in-place and returns it.
    """
    # Find all simple cycles in the graph
    loops = nx.cycle_basis(G)
    for cycle in loops:
        if len(cycle) != 3:
            continue

        # Identify the heaviest edge in the triangle
        max_edge = None
        max_weight = float('-inf')
        for i in range(3):
            u = cycle[i]
            v = cycle[(i + 1) % 3]

            # safely get weight from either direction
            if G.has_edge(u, v):
                w = G[u][v].get('weight', 0)
            elif G.has_edge(v, u):
                w = G[v][u].get('weight', 0)
            else:
                # no such edge—skip
                continue

            if w > max_weight:
                max_weight = w
                max_edge = (u, v)

        # Remove the heaviest edge if it still exists
        if max_edge:
            u, v = max_edge
            if G.has_edge(u, v):
                G.remove_edge(u, v)
            elif G.has_edge(v, u):
                G.remove_edge(v, u)

    return G


def save_results(results, output_folder, save_segment_masks, save_conn_comp_masks):
    """Save computed metrics to organized folder structure with CSVs."""
    
    general_keys = [
        'total_length', 'num_bifurcations', 'bifurcation_density', 'volume',
        'fractal_dimension', 'lacunarity',
        'num_loops', 'num_abnormal_degree_nodes'
    ]

    all_rows = []
    region_rows = []

    # Extract regions
    if 'region_metrics' in results:
        region_results = results['region_metrics']
    else:
        region_results = {'global': results}

    for region_label, region_data in region_results.items():
        if not isinstance(region_data, dict):
            continue  # safety
        components = {
            k: v for k, v in region_data.items()
            if isinstance(v, dict) and 'total_length' in v
        }

        # Sort by length
        sorted_items = sorted(
            components.items(),
            key=lambda item: item[1].get('total_length', 0),
            reverse=True
        )

        for new_idx, (cid, data) in enumerate(sorted_items):
            comp_idx = new_idx + 1
            region_name = f"Region_{region_label}"
            comp_dir = os.path.join(output_folder, region_name, f"Conn_comp_{comp_idx}")
            os.makedirs(comp_dir, exist_ok=True)

            row = {
                'region_label': region_label,
                'component_index': comp_idx,
                'original_component_id': cid
            }
            for k in general_keys:
                row[k] = data.get(k, np.nan)

            # Tortuosity
            agg = data.get('aggregated_tortuosity_by_root', [])
            prefixes = [
                'Largest_endpoint_root_',
                '2nd_Largest_endpoint_root_',
                'Largest_bifurcation_root_'
            ]
            for i, prefix in enumerate(prefixes):
                if i < len(agg):
                    row[prefix + 'mean_curvature'] = agg[i].get('mean_curvature', np.nan)
                    row[prefix + 'mean_square_curvature'] = agg[i].get('mean_square_curvature', np.nan)
                else:
                    row[prefix + 'mean_curvature'] = np.nan
                    row[prefix + 'mean_square_curvature'] = np.nan

            all_rows.append({k: v for k, v in row.items() if k != 'region_label'})
            region_rows.append(row)

            # Save skeleton
            if save_conn_comp_masks and 'reconstructed_conn_comp' in data:
                nib.save(
                    data['reconstructed_conn_comp'],
                    os.path.join(comp_dir, f'Conn_comp_{comp_idx}_skeleton.nii.gz')
                )

            # Save segments
            if 'segments_by_root' in data:
                segs_dir = os.path.join(comp_dir, 'Segments')
                os.makedirs(segs_dir, exist_ok=True)
                root_names = [
                    'Largest endpoint root',
                    'Second largest endpoint root',
                    'Largest bifurcation root'
                ]
                for root_idx, root_entry in enumerate(data['segments_by_root']):
                    if root_idx >= len(root_names):
                        break

                    root_dir = os.path.join(segs_dir, root_names[root_idx])
                    os.makedirs(root_dir, exist_ok=True)

                    metrics_list = root_entry.get('segment_metrics', [])
                    masks_list = root_entry.get('segment_masks', []) if save_segment_masks else []

                    for seg_idx, sm in enumerate(metrics_list, start=1):
                        seg_dir = os.path.join(root_dir, f"Segment_{seg_idx}")
                        os.makedirs(seg_dir, exist_ok=True)

                        pd.DataFrame([sm]).to_csv(
                            os.path.join(seg_dir, 'Segment_metrics.csv'), index=False
                        )

                        if save_segment_masks and seg_idx - 1 < len(masks_list):
                            nib.save(
                                masks_list[seg_idx - 1],
                                os.path.join(seg_dir, 'Segment.nii.gz')
                            )

    # === Per-region/component CSV ===
    if region_rows:
        df_detailed = pd.DataFrame(region_rows)
        df_detailed = df_detailed.drop(columns=['component_index', 'original_component_id'], errors='ignore')
        df_detailed = df_detailed.sort_values(
            by=['region_label', 'total_length'],
            ascending=[True, False]
        )
        df_detailed.to_csv(
            os.path.join(output_folder, 'all_components_by_region.csv'),
            index=False
        )

        # === Region Summary ===
        summary_metrics = [
            'total_length', 'num_bifurcations', 'volume', 'num_loops', 'num_abnormal_degree_nodes'
        ]

        tortuosity_cols = [
            'Largest_endpoint_root_mean_curvature',
            'Largest_endpoint_root_mean_square_curvature',
            '2nd_Largest_endpoint_root_mean_curvature',
            '2nd_Largest_endpoint_root_mean_square_curvature',
            'Largest_bifurcation_root_mean_curvature',
            'Largest_bifurcation_root_mean_square_curvature'
        ]

        region_groups = df_detailed.groupby('region_label')
        summary_rows = []

        for region_label, group in region_groups:
            row = {
                'region_label': region_label,
                'num_components': len(group)
            }

            for metric in summary_metrics:
                row[metric] = group[metric].sum()

            total_len = group['total_length'].sum()
            for col in tortuosity_cols:
                if col in group:
                    values = group[col].fillna(0)
                    weighted_avg = (values * group['total_length']).sum() / total_len if total_len > 0 else np.nan
                    row[col] = weighted_avg
                else:
                    row[col] = np.nan

            region_data = region_results.get(region_label, {})
            for b in ['betti_0', 'betti_1', 'betti_2']:
                row[b] = region_data.get(b, np.nan)

            summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows)
        df_summary = df_summary.sort_values(by='region_label')
        df_summary.to_csv(
            os.path.join(output_folder, 'region_summary.csv'),
            index=False
        )

    print(f"Results saved to: {output_folder}")

    
def compute_metrics_for_mask(mask, dist_map, affine, header, selected_metrics, 
                              save_segment_masks, save_conn_comp_masks):
    """Compute all requested metrics for a given mask region."""
    
    skel = skeletonize(mask)
    G = prune_graph(build_graph(skel))
    results = {}
    
    for cid, comp_nodes in enumerate(nx.connected_components(G)):
        Gc = G.subgraph(comp_nodes)
        data = {}

        if save_conn_comp_masks:
            vessel_mask = np.zeros_like(skel, dtype=bool)
            for node in Gc.nodes:
                r = dist_map[node]
                if r > 0:
                    rr = int(np.ceil(r))
                    x0, y0, z0 = node
                    for dx in range(-rr, rr + 1):
                        for dy in range(-rr, rr + 1):
                            for dz in range(-rr, rr + 1):
                                x, y, z = x0 + dx, y0 + dy, z0 + dz
                                if (0 <= x < vessel_mask.shape[0] and
                                    0 <= y < vessel_mask.shape[1] and
                                    0 <= z < vessel_mask.shape[2]):
                                    if np.sqrt(dx**2 + dy**2 + dz**2) <= r:
                                        vessel_mask[x, y, z] = True
            data['reconstructed_conn_comp'] = nib.Nifti1Image(vessel_mask.astype(np.uint8), affine, header)

        coords = np.array(list(Gc.nodes()))
        total_len = sum(d['weight'] for *_, d in Gc.edges(data=True))
        num_bif = sum(1 for _, deg in Gc.degree() if deg >= 3)

        if 'num_bifurcations' in selected_metrics:
            data['num_bifurcations'] = num_bif
        if 'total_length' in selected_metrics:
            data['total_length'] = total_len
        if 'bifurcation_density' in selected_metrics:
            data['bifurcation_density'] = num_bif / total_len if total_len > 0 else np.nan
        if 'volume' in selected_metrics:
            vol = 0.0
            for u, v, d in Gc.edges(data=True):
                r_avg = (dist_map[u] + dist_map[v]) / 2
                vol += np.pi * (r_avg**2) * d['weight']
            data['volume'] = vol
        if 'fractal_dimension' in selected_metrics:
            data['fractal_dimension'] = fractal_dimension(coords)
        if 'lacunarity' in selected_metrics:
            box_size = np.max(coords.max(axis=0) - coords.min(axis=0)) / 10 or 1
            data['lacunarity'] = calculate_lacunarity(coords, box_size)
        if 'num_loops' in selected_metrics or 'num_abnormal_degree_nodes' in selected_metrics:
            nl, nab = analyze_component_structure(Gc)
            if 'num_loops' in selected_metrics:
                data['num_loops'] = nl
            if 'num_abnormal_degree_nodes' in selected_metrics:
                data['num_abnormal_degree_nodes'] = nab

        if any(m in selected_metrics for m in [
            'geodesic_length', 'avg_diameter', 'spline_mean_curvature', 'spline_mean_square_curvature'
        ]):
            seg_lists, seg_counts = extract_segments_from_component_using_shortest_path(Gc, dist_map)
            segments_info = []
            agg_curv = []
            agg_curv2 = []

            for r_idx, seg_list in enumerate(seg_lists):
                seg_metrics = []
                seg_masks = []
                total_geo = 0.0
                sum_curv = 0.0
                sum_curv2 = 0.0

                for seg in seg_list:
                    pts = np.array(seg)
                    sm = {}

                    L_geo = sum(np.linalg.norm(pts[i]-pts[i+1]) for i in range(len(pts)-1))
                    sm['geodesic_length'] = L_geo
                    total_geo += L_geo

                    if 'avg_diameter' in selected_metrics:
                        sm['avg_diameter'] = np.mean([2*dist_map[tuple(p)] for p in pts])

                    counts_arr = np.array([seg_counts[r_idx].get(tuple(p),1) for p in pts])
                    tort = compute_tortuosity_metrics(pts, smoothing=0, n_samples=500, counts=counts_arr)
                    sm.update(tort)

                    if 'spline_mean_curvature' in selected_metrics:
                        sm['spline_mean_curvature'] = tort['spline_mean_curvature']
                        sum_curv += tort['spline_mean_curvature'] * L_geo
                    if 'spline_mean_square_curvature' in selected_metrics:
                        sm['spline_mean_square_curvature'] = tort['spline_mean_square_curvature']
                        sum_curv2 += tort['spline_mean_square_curvature'] * L_geo

                    if save_segment_masks:
                        mask_i = np.zeros_like(skel, dtype=bool)
                        for node in seg:
                            r = dist_map[tuple(node)]
                            if r <= 0: continue
                            rr = int(np.ceil(r)); x0,y0,z0 = node
                            for dx in range(-rr,rr+1):
                                for dy in range(-rr,rr+1):
                                    for dz in range(-rr,rr+1):
                                        xi, yi, zi = x0+dx, y0+dy, z0+dz
                                        if (0 <= xi < mask_i.shape[0] and
                                            0 <= yi < mask_i.shape[1] and
                                            0 <= zi < mask_i.shape[2] and
                                            dx*dx + dy*dy + dz*dz <= r*r):
                                            mask_i[xi, yi, zi] = True
                        seg_masks.append(nib.Nifti1Image(mask_i.astype(np.uint8), affine, header))

                    seg_metrics.append(sm)

                if total_geo > 0:
                    agg_curv.append(sum_curv/total_geo)
                    agg_curv2.append(sum_curv2/total_geo)
                else:
                    agg_curv.append(np.nan)
                    agg_curv2.append(np.nan)

                segments_info.append({
                    'segment_metrics': seg_metrics,
                    'segment_masks': seg_masks
                })

            data['segments_by_root'] = segments_info
            data['aggregated_tortuosity_by_root'] = [
                {'mean_curvature': agg_curv[i],
                 'mean_square_curvature': agg_curv2[i]}
                for i in range(len(agg_curv))
            ]

        results[cid] = data

    return results


def register_atlas_to_segmentation(atlas_path, segmentation_path, output_registered_atlas_path,
                                   mask_atlas=None, mask_segmentation=None):
    """
    Register an atlas image onto a segmentation image using FLIRT (rigid, mutual info),
    saving only the registered atlas image.

    Args:
        atlas_path (str): Path to the atlas image to be registered.
        segmentation_path (str): Path to the segmentation image (reference).
        output_registered_atlas_path (str): Path to save the registered atlas image.
        mask_atlas (str, optional): Mask path for atlas image.
        mask_segmentation (str, optional): Mask path for segmentation image.
    """
    print('Starting atlas registration...')
    dirname, _ = os.path.split(output_registered_atlas_path)
    TEMP_DIRECTORY = os.path.join(dirname, "tmp")
    os.makedirs(TEMP_DIRECTORY, exist_ok=True)

    flirt = fsl.FLIRT()
    flirt.inputs.in_file = atlas_path
    flirt.inputs.reference = segmentation_path
    flirt.inputs.out_file = output_registered_atlas_path
    flirt.inputs.out_matrix_file = os.path.join(
        TEMP_DIRECTORY,
        "FLIRT_transform-matrix.mat"
    )

    flirt.inputs.cost = "mutualinfo"
    flirt.inputs.dof = 12  # rigid-body

    if mask_atlas and mask_segmentation:
        flirt.inputs.in_weight = mask_atlas
        flirt.inputs.ref_weight = mask_segmentation
    
    flirt.run()
    print('Atlas registration completed.')


def process(image_path, mask_path, selected_metrics, save_segment_masks, save_conn_comp_masks, 
            image_T1=None, atlas_path=None):
    """
    Main processing function for vessel metrics computation.
    
    Supports two modes:
    1. Whole-brain analysis: Only image_path and mask_path provided
    2. Atlas-based regional analysis: image_path, mask_path, image_T1, and atlas_path provided
    
    Args:
        image_path (str): Path to vessel image (.nii/.nii.gz)
        mask_path (str): Path to vessel mask (.nii/.nii.gz)
        selected_metrics (set): Set of metrics to compute
        save_segment_masks (bool): Whether to save segment masks
        save_conn_comp_masks (bool): Whether to save connected component masks
        image_T1 (str, optional): Path to T1 image for atlas registration
        atlas_path (str, optional): Path to atlas file for regional parcellation
    
    Returns:
        dict: Results dictionary containing regional or whole-brain metrics
    """
    # Load segmentation mask
    seg = nib.load(mask_path)
    arr_seg = seg.get_fdata()
    affine_seg, header_seg = seg.affine, seg.header

    # Round and cast to integer (binarize)
    arr_seg = np.round(arr_seg).astype(int)
    
    # Distance map for the segmentation mask
    dist_map = distance_transform_edt(arr_seg)

    # Determine analysis mode
    use_atlas = (atlas_path is not None and image_T1 is not None)
    
    if use_atlas:
        # ATLAS-BASED REGIONAL ANALYSIS MODE
        print("="*60)
        print("ATLAS-BASED REGIONAL ANALYSIS MODE")
        print("="*60)
        
        image_name = os.path.basename(image_path).split('.')[0]
        image_T1_name = os.path.basename(image_T1).split('.')[0]
        
        # TEMP paths for registration outputs
        reg_atlas_path = os.path.join(os.path.dirname(mask_path), f"{image_name}_registered_atlas.nii.gz")
        reg_atlas_T1_path = os.path.join(os.path.dirname(mask_path), f"{image_T1_name}_registered_atlas.nii.gz")
        reg_T1_path = os.path.join(os.path.dirname(mask_path), f"{image_T1_name}_registered_T1.nii.gz")
        cropped_path = os.path.join(os.path.dirname(mask_path), f"{image_T1_name}_cropped.nii.gz")
        
        # Preprocessing: Crop T1 image
        print('Cropping T1 image...')
        fslroi = fsl.ExtractROI()
        fslroi.inputs.in_file = image_T1
        fslroi.inputs.t_min = 0
        fslroi.inputs.t_size = -1
        fslroi.inputs.x_min = 0
        fslroi.inputs.x_size = -1
        fslroi.inputs.y_min = 50
        fslroi.inputs.y_size = -1
        fslroi.inputs.z_min = 0
        fslroi.inputs.z_size = -1
        
        result = fslroi.run()
        cropped_file = result.outputs.roi_file
        import shutil
        shutil.move(cropped_file, cropped_path)
        print(f"  → Cropped volume saved to {cropped_path}")
        
        # Extract brain mask
        print('Extracting brain mask...')
        bet = fsl.BET()
        bet.inputs.in_file = cropped_path
        bet.inputs.mask = True
        bet.inputs.frac = 0.30
        bet.inputs.vertical_gradient = 0.0
        bet.inputs.out_file = os.path.join(os.path.dirname(mask_path), f"{image_T1_name}_brain.nii.gz")
        image_path_no_brain = os.path.join(os.path.dirname(mask_path), f"{image_T1_name}_brain.nii.gz")
        bet.run()
        print(f"  → Brain extracted to {image_path_no_brain}")
        
        # Reorient to standard
        print('Reorienting to standard space...')
        reorient = fsl.Reorient2Std()
        reorient.inputs.in_file = image_path_no_brain
        reorient.inputs.out_file = image_path_no_brain
        reorient.run()
        
        # Register atlas to T1
        print('Registering atlas to T1...')
        register_atlas_to_segmentation(
            atlas_path=atlas_path,
            segmentation_path=image_path_no_brain,
            output_registered_atlas_path=reg_atlas_T1_path
        )
        
        # Register T1 to MRA
        print('Registering T1 to MRA...')
        flirt = fsl.FLIRT()
        flirt.inputs.in_file = image_path_no_brain
        flirt.inputs.reference = image_path
        flirt.inputs.out_file = reg_T1_path
        flirt.inputs.interp = 'trilinear'
        flirt.inputs.apply_xfm = True
        flirt.inputs.uses_qform = True
        flirt.run()
        
        # Register atlas to MRA
        print('Registering atlas to MRA...')
        flirt = fsl.FLIRT()
        flirt.inputs.in_file = reg_atlas_T1_path
        flirt.inputs.reference = image_path
        flirt.inputs.out_file = reg_atlas_path
        flirt.inputs.interp = 'nearestneighbour'
        flirt.inputs.apply_xfm = True
        flirt.inputs.uses_qform = True
        flirt.run()
        
        # Load registered atlas
        atlas_nib = nib.load(reg_atlas_path)
        atlas = np.round(atlas_nib.get_fdata()).astype(int)
        
        # Compute metrics per region
        results = {'region_metrics': {}}
        unique_labels = np.unique(atlas)
        unique_labels = unique_labels[unique_labels != 0]  # exclude background
        
        print(f"\nProcessing {len(unique_labels)} atlas regions...")
        for idx, label in enumerate(unique_labels, 1):
            region_mask = (atlas == label) & (arr_seg == 1)
            
            if not np.any(region_mask):
                print(f'  [{idx}/{len(unique_labels)}] Region {label}: No vessels found')
                continue
            
            print(f'  [{idx}/{len(unique_labels)}] Processing region {label}...')
            region_result = compute_metrics_for_mask(
                region_mask, dist_map, affine_seg, header_seg, 
                selected_metrics, save_segment_masks, save_conn_comp_masks
            )
            results['region_metrics'][int(label)] = region_result
        
        print(f"\nAtlas-based analysis complete. Processed {len(results['region_metrics'])} regions.")
    
    else:
        # WHOLE-BRAIN ANALYSIS MODE (single region)
        print("="*60)
        print("WHOLE-BRAIN ANALYSIS MODE (Single Region)")
        print("="*60)
        
        # Treat entire segmentation as a single region (label = 1)
        results = {'region_metrics': {}}
        
        # Create binary mask (all vessel voxels belong to region 1)
        region_mask = (arr_seg >= 1)
        
        if not np.any(region_mask):
            print('ERROR: No segmentation found in mask')
            return results
        
        num_voxels = np.sum(region_mask)
        print(f'Processing whole-brain vessel network ({num_voxels:,} voxels)...')
        
        region_result = compute_metrics_for_mask(
            region_mask, dist_map, affine_seg, header_seg,
            selected_metrics, save_segment_masks, save_conn_comp_masks
        )
        results['region_metrics'][1] = region_result  # Label as region 1
        
        print("Whole-brain analysis complete.")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Vessel metrics computation (whole-brain or atlas-based regional analysis)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Whole-brain analysis:
  python %(prog)s image.nii.gz mask.nii.gz
  
  # Atlas-based regional analysis:
  python %(prog)s image.nii.gz mask.nii.gz --image_T1 T1.nii.gz --atlas_path atlas.nii.gz
        """
    )
    
    # Required arguments
    parser.add_argument('image_path', help='Path to vessel image (.nii/.nii.gz)')
    parser.add_argument('mask_path', help='Path to vessel mask (.nii/.nii.gz)')
    
    # Optional arguments for atlas-based mode
    parser.add_argument('--image_T1', default=None,
                       help='Path to T1 image (.nii/.nii.gz) for atlas registration')
    parser.add_argument('--atlas_path', default=None,
                       help='Path to atlas file (.nii/.nii.gz) for regional parcellation')
    
    # Analysis options
    parser.add_argument('--metrics', nargs='+', default=None,
                       help='Metrics to compute (default: all)')
    parser.add_argument('--output_folder', default='./VESSEL_METRICS',
                       help='Save directory (default: ./VESSEL_METRICS)')
    parser.add_argument('--no_segment_masks', action='store_true',
                       help='Disable segment mask construction and saving')
    parser.add_argument('--no_conn_comp_masks', action='store_true',
                       help='Disable connected component reconstruction saving')

    args = parser.parse_args()

    # All available metrics
    all_keys = [
        'total_length', 'num_bifurcations', 'bifurcation_density', 'volume',
        'fractal_dimension', 'lacunarity', 'geodesic_length', 'avg_diameter',
        'spline_arc_length', 'spline_chord_length', 'spline_mean_curvature',
        'spline_mean_square_curvature', 'spline_rms_curvature', 'arc_over_chord',
        'fit_rmse', 'num_loops', 'num_abnormal_degree_nodes'
    ]

    # Parse selected metrics
    selected_metrics = set(args.metrics) if args.metrics else set(all_keys)
    invalid = selected_metrics - set(all_keys)
    if invalid:
        raise ValueError(f"Invalid metrics requested: {invalid}")

    # Validate atlas mode requirements
    if (args.image_T1 is None) != (args.atlas_path is None):
        parser.error("For atlas-based analysis, both --image_T1 and --atlas_path must be provided")

    # Run pipeline
    save_segment_masks = not args.no_segment_masks
    save_conn_comp_masks = not args.no_conn_comp_masks

    print("\nVessel Analysis Pipeline")
    print("="*60)
    print(f"Image:               {args.image_path}")
    print(f"Mask:                {args.mask_path}")
    if args.image_T1 and args.atlas_path:
        print(f"T1 Image:            {args.image_T1}")
        print(f"Atlas:               {args.atlas_path}")
        print(f"Mode:                Atlas-based regional analysis")
    else:
        print(f"Mode:                Whole-brain analysis")
    print(f"Output folder:       {args.output_folder}")
    print(f"Metrics:             {len(selected_metrics)} selected")
    print(f"Save segment masks:  {save_segment_masks}")
    print(f"Save component masks: {save_conn_comp_masks}")
    print("="*60)
    print()

    results = process(
        image_path=args.image_path,
        mask_path=args.mask_path,
        selected_metrics=selected_metrics,
        save_segment_masks=save_segment_masks,
        save_conn_comp_masks=save_conn_comp_masks,
        image_T1=args.image_T1,
        atlas_path=args.atlas_path
    )

    save_results(results, args.output_folder, save_segment_masks, save_conn_comp_masks)

    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"Results saved to: {os.path.abspath(args.output_folder)}")
    print(f"{'='*60}\n")