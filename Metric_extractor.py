# -*- coding: utf-8 -*-
"""
vessel_analysis_modular.py

A command-line tool to analyze 3D vessel masks by skeletonizing,
building a graph, extracting segments, and computing only requested metrics.

USAGE:
    python vessel_analysis_modular.py <input_path> [--metrics METRIC [METRIC ...]] [--output_folder PATH]

ARGUMENTS:
    input_path       Path to the NIfTI vessel mask (.nii or .nii.gz) or to the .npy.
    --metrics        (optional) List of metrics to compute/display. Options include:
                     total_length, bifurcation_density, volume, fractal_dimension,
                     lacunarity, geodesic_length, avg_diameter,
                     spline_arc_length, spline_chord_length, mean_curvature,
                     mean_square_curvature, rms_curvature, arc_over_chord, rmse.
                     Default is all metrics.
    --output_folder  (optional) Path to save results. Default is './VESSEL METRICS'.
"""
import argparse
import os
from collections import defaultdict

import networkx as nx
import nibabel as nib
import numpy as np
import pandas as pd
from nipype.interfaces import fsl
from scipy.interpolate import splprep, splev, interp1d
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from sklearn.linear_model import LinearRegression
import pickle
import SimpleITK as sitk
import vedo
from pathlib import Path


def load_label_map(txt_path):
    labels = {}
    if not os.path.exists(txt_path): return labels
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                labels[int(parts[0])] = " ".join(parts[1:])
    return labels

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
    try:
        mean_curv       = np.trapz(curv_w,  s_uniform)
        mean_sq_curv    = np.trapz(curv2_w, s_uniform)
    except Exception as e:
        mean_curv    = np.trapezoid(curv_w,  s_uniform)
        mean_sq_curv = np.trapezoid(curv2_w, s_uniform)
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


def compute_fractal_dimension(points, box_sizes=None):
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
    diameters_eps = {node: 2 * distance_map.get(node, 0.5) for node in endpoints}
    # Sort endpoints by diameter descending
    sorted_eps = sorted(endpoints, key=lambda n: diameters_eps[n], reverse=True)

    # First two roots: largest and second-largest diameter endpoints
    root1 = sorted_eps[0]
    root2 = sorted_eps[1] if len(sorted_eps) > 1 else root1

    # Third root: bifurcation (degree >= 3) with largest diameter, or fallback
    bif_nodes = [node for node, deg in G_comp.degree() if deg >= 3]
    if bif_nodes:
        diameters_bif = {node: 2 * distance_map.get(node, 0.5) for node in bif_nodes}
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

def save_results(results, output_folder, save_segment_masks, save_conn_comp_masks):

    print("  Saving results...")
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
        print("  No regional metrics found, saving global results only.")
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
            comp_dir = os.path.join(output_folder, "regional_results" ,region_name, f"Conn_comp_{comp_idx}")
            Path(comp_dir).mkdir(parents=True, exist_ok=True)

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
            if save_conn_comp_masks:
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
        print("  Saving detailed per-region/component metrics...")
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

    
    
def compute_metrics_for_mask(G, node_radius_map, selected_metrics, save_segment_masks=None, save_conn_comp_masks=None):
    results = {}
    for cid, comp_nodes in enumerate(nx.connected_components(G)):
        Gc = G.subgraph(comp_nodes).copy()
        data = {}
        

        coords = np.array([Gc.nodes[n]['pos'] for n in Gc.nodes()])
        total_len = Gc.size(weight='weight')
        degrees = dict(Gc.degree())
        num_bif = sum(1 for d in degrees.values() if d >= 3)

        
        if 'num_bifurcations' in selected_metrics:
            data['num_bifurcations'] = num_bif
        
        if 'total_length' in selected_metrics:
            data['total_length'] = total_len
        
        if 'bifurcation_density' in selected_metrics:
            data['bifurcation_density'] = num_bif / total_len if total_len > 0 else np.nan
        
        if 'volume' in selected_metrics:
            vol = 0.0    
            for u, v, d in Gc.edges(data=True):
                l = d.get('weight', 0)
                r = (node_radius_map.get(u, 0.5) + node_radius_map.get(v, 0.5)) / 2.0
                vol += l * np.pi * (r ** 2) 
            data['volume'] = vol
        
        if 'fractal_dimension' in selected_metrics:
            data['fractal_dimension'] = compute_fractal_dimension(coords)
        
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
            seg_lists, seg_counts = extract_segments_from_component_using_shortest_path(Gc, node_radius_map)
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
                    pts = np.array([Gc.nodes[n]['pos'] for n in seg])
                    sm = {}

                    L_geo = sum(np.linalg.norm(pts[i] - pts[i+1]) for i in range(len(pts)-1))
                    sm['geodesic_length'] = L_geo
                    total_geo += L_geo

                    if 'avg_diameter' in selected_metrics:
                        sm['avg_diameter'] = np.mean([2 * node_radius_map.get(n, 0.5) for n in seg])
                        
                    counts_arr = np.array([seg_counts[r_idx].get(n, 1) for n in seg])
                    tort = compute_tortuosity_metrics(pts, smoothing=0, n_samples=500, counts=counts_arr)
                    sm.update(tort)

                    if 'spline_mean_curvature' in selected_metrics:
                        sm['spline_mean_curvature'] = tort['spline_mean_curvature']
                        # Only add to total if value is valid (not NaN)
                        if not np.isnan(tort['spline_mean_curvature']):
                            sum_curv += tort['spline_mean_curvature'] * L_geo
                            
                    if 'spline_mean_square_curvature' in selected_metrics:
                        sm['spline_mean_square_curvature'] = tort['spline_mean_square_curvature']
                        # Only add to total if value is valid
                        if not np.isnan(tort['spline_mean_square_curvature']):
                            sum_curv2 += tort['spline_mean_square_curvature'] * L_geo
                            
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

def extract_metrics(patient_name, output_folder, graph_pkl_path, selected_metrics, label_map_path, registered_atlas_path=None, save_segment_masks=None, save_conn_comp_masks=None):
    """
    Main processing function with atlas registration and metric computation.
    """
    # ============================================================================
    # 6. Compute the vessel metrics
    # ============================================================================
    use_atlas = True if registered_atlas_path else False
    results = {'region_metrics': {}} if use_atlas else {}
    
    if use_atlas:
        aligned_atlas = sitk.ReadImage(registered_atlas_path)
    
    # ============================================================================
    # Load the graph pickle
    # ============================================================================    
    try:
        import sys
        if 'numpy._core' not in sys.modules:
            try:
                import numpy.core as _core
                sys.modules['numpy._core'] = _core
            except: pass
        with open(graph_pkl_path, 'rb') as f:
            try:
                data = pickle.load(f, encoding='latin1')
            except:
                f.seek(0)
                data = pickle.load(f)
        G = data['graph']
        node_radius_map = data['node_radius_map']
    except Exception as e:
        print(f"  ERROR: Failed to load pickle - {e}")
        return
    
    label_map = load_label_map(label_map_path)
    
    # ============================================================================
    # Atlas-based regional analysis
    # ============================================================================
    if use_atlas:
        # Map Nodes to Regions
        region_nodes = {}
        size = aligned_atlas.GetSize()
        for n in G.nodes():
            pos = G.nodes[n]['pos']
            try:
                idx = aligned_atlas.TransformPhysicalPointToIndex(pos)
                if (0 <= idx[0] < size[0] and 0 <= idx[1] < size[1] and 0 <= idx[2] < size[2]):
                    region_id = aligned_atlas[idx]
                    if region_id > 0:
                        if region_id not in region_nodes: region_nodes[region_id] = []
                        region_nodes[region_id].append(n)
            except:
                # Raise error if index transformation fails
                raise ValueError(f"Failed to transform point {pos} to index in atlas.")
            
        for r_id in sorted(region_nodes.keys()):
            r_name = label_map.get(r_id, f"{r_id}")
            #print(f"  Computing metrics for region {r_id} with {len(region_nodes[r_id])} nodes")
            sub_G = G.subgraph(region_nodes[r_id]).copy()
            
            region_result = compute_metrics_for_mask(
                sub_G, node_radius_map,selected_metrics, save_segment_masks=None, save_conn_comp_masks=None
            )
            results['region_metrics'][int(r_id)] = region_result
        
        save_results(results, output_folder, save_segment_masks, save_conn_comp_masks)
    
        # Saving regional VTP
        n_regions = len(region_nodes)
        print(f"  Found {n_regions} regions. Generating regional VTP for {patient_name}...")
        regional_actors = []
        
        # Create directory for individual region VTPs
        regions_vtp_dir = os.path.join(output_folder, "regional_vtps")
        os.makedirs(regions_vtp_dir, exist_ok=True)
        
        for r_id, nodes in region_nodes.items():
            sub_G = G.subgraph(nodes)
            
            # Extract edges within this region
            s_pts = [sub_G.nodes[u]['pos'] for u, v in sub_G.edges()]
            e_pts = [sub_G.nodes[v]['pos'] for u, v in sub_G.edges()]
            
            if s_pts:
                # Create the lines
                region_lines = vedo.Lines(s_pts, e_pts).lw(3)
                
                # Create an array of the Region ID for every cell (edge)
                n_cells = region_lines.dataset.GetNumberOfCells()
                visual_id = (int(r_id) * 137) % 256  # Simple color hash
                region_array = np.full(n_cells, visual_id, dtype=np.int32)
                
                # Add this array to the cell data so Slicer can map colors to IDs
                region_lines.celldata["RegionID"] = region_array
                regional_actors.append(region_lines)
                
                # Save individual region VTP
                region_name = label_map.get(r_id, f"region_{r_id}")
                individual_vtp_path = os.path.join(regions_vtp_dir, f"region_{r_id}_{region_name}.vtp")
                region_lines.dataset.GetCellData().SetActiveScalars("RegionID")
                region_lines.write(individual_vtp_path)
                print(f"  -> Saved individual VTP for region {r_id} ({region_name}): {individual_vtp_path}")

        if regional_actors:
            # Merge all regional line sets into one object
            combined_vtp = vedo.merge(regional_actors)
            regional_vtp_path = os.path.join(output_folder, "vessel_graph_regional.vtp")
            
            # Explicitly set RegionID as the active scalar for Slicer to read
            combined_vtp.dataset.GetCellData().SetActiveScalars("RegionID")
            combined_vtp.write(regional_vtp_path)
            print(f"  -> Saved combined VTP with {n_regions} regions to: {regional_vtp_path}")
    
    # ============================================================================    
    # NOT using atlas
    # ============================================================================    
    else:
        print("  Computing global vessel metrics...")
        
        # Create a Results_GLOBAL folder
        parent, child = os.path.split(output_folder.rstrip('/'))
        output_folder = os.path.join(parent + "_GLOBAL", child)
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        global_result = compute_metrics_for_mask(
            G, node_radius_map, selected_metrics, save_segment_masks=None, save_conn_comp_masks=None
        )
        results.update(global_result)
        save_results(results, output_folder, save_segment_masks, save_conn_comp_masks)
        
