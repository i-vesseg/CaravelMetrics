# -*- coding: utf-8 -*-
"""
Metric_extractor.py

A command-line tool to analyze 3D vessel masks by skeletonizing,
building a graph, extracting segments, and computing requested metrics.
"""
import argparse
import os
from collections import defaultdict
import networkx as nx
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev, interp1d
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression
import pickle
import SimpleITK as sitk
import vedo
from pathlib import Path


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

def compute_tortuosity_metrics(points, smoothing=0, n_samples=500, counts=None):
    pts = np.asarray(points)
    
    # Consecutive duplicate points crash splprep; drop them (keep the first).
    if pts.shape[0] > 1:
        diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        valid_idx = np.insert(diffs > 1e-6, 0, True)
        pts = pts[valid_idx]
        if counts is not None:
            counts = np.asarray(counts)[valid_idx]

    if pts.shape[0] < 4:
        nan_dict = {k: np.nan for k in [
            'spline_arc_length','spline_chord_length',
            'spline_mean_curvature','spline_mean_square_curvature',
            'spline_rms_curvature','arc_over_chord']}
        return nan_dict

    tck, u = splprep(pts.T, s=smoothing)
    u_fine   = np.linspace(0, 1, n_samples)
    deriv1   = np.array(splev(u_fine, tck, der=1)).T
    du       = np.gradient(u_fine)
    ds       = np.linalg.norm(deriv1, axis=1) * du
    s_cum    = np.cumsum(ds) - ds[0]
    arc_len  = s_cum[-1]

    u_of_s   = interp1d(s_cum, u_fine, kind='linear', bounds_error=False, fill_value=(0,1))
    s_uniform= np.linspace(0, arc_len, n_samples)
    u_uniform= u_of_s(s_uniform)
    pts_u    = np.array(splev(u_uniform, tck)).T

    dt  = s_uniform[1] - s_uniform[0]
    d1  = np.gradient(pts_u, dt, axis=0)
    d2  = np.gradient(d1, dt, axis=0)

    if counts is not None:
        counts_orig  = np.asarray(counts)
        interp_cnt   = interp1d(u, counts_orig, kind='linear', bounds_error=False, fill_value=(counts_orig[0], counts_orig[-1]))
        n_s          = interp_cnt(u_uniform)
        n_s          = np.where(n_s <= 0, 1, n_s) 
    else:
        n_s = np.ones(n_samples)

    cross_vec = np.cross(d1, d2)
    speed     = np.linalg.norm(d1, axis=1)
    curvature = np.linalg.norm(cross_vec, axis=1) / (speed**3 + 1e-10)

    curv_w    = curvature / n_s
    curv2_w   = (curvature**2) / (n_s**2)

    # Correct Integral Calculation
    try:
        integral_curv    = np.trapz(curv_w,  s_uniform)
        integral_sq_curv = np.trapz(curv2_w, s_uniform)
    except Exception as e:
        integral_curv    = np.trapezoid(curv_w,  s_uniform)
        integral_sq_curv = np.trapezoid(curv2_w, s_uniform)
        
    mean_curv       = (integral_curv / arc_len) if arc_len > 0 else 0
    mean_sq_curv    = (integral_sq_curv / arc_len) if arc_len > 0 else 0
    rms_curv        = np.sqrt(mean_sq_curv)

    chord_len = np.linalg.norm(pts_u[-1] - pts_u[0])
    # 'fit_rmse' (spline vs input points) is deliberately not reported: with
    # s=0 (exact interpolation, used throughout) it is ~1e-14 everywhere --
    # floating-point noise, not a meaningful fit-quality column.

    return {
        'spline_arc_length':           arc_len,
        'spline_chord_length':         chord_len,
        'spline_mean_curvature':       mean_curv,
        'spline_mean_square_curvature':mean_sq_curv,
        'spline_rms_curvature':        rms_curv,
        'arc_over_chord':              (arc_len / chord_len) if chord_len>0 else np.inf,
    }

def compute_fractal_dimension(points, box_sizes=None):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return np.nan

    mins = points.min(axis=0)
    shifted = points - mins
    maxs = shifted.max(axis=0)
    max_dim = max(maxs)

    if max_dim == 0: return np.nan

    if box_sizes is None:
        box_sizes = np.logspace(np.log10(max_dim / 50.0), np.log10(max_dim), num=10, base=10.0)
    
    counts = []
    valid_box_sizes = []
    for size in box_sizes:
        if size <= 0 or np.isnan(size): continue
        bins = np.ceil(maxs / size).astype(int) + 1
        indices = np.floor(shifted / size).astype(int)
        unique_boxes = {tuple(idx) for idx in indices}
        counts.append(len(unique_boxes))
        valid_box_sizes.append(size)
    
    if len(counts) == 0: return np.nan

    X = np.log(1.0 / np.array(valid_box_sizes)).reshape(-1, 1)
    y = np.log(counts)
    reg = LinearRegression().fit(X, y)
    return reg.coef_[0]

def calculate_lacunarity(points, box_size):
    points = np.asarray(points)
    mins = points.min(axis=0)
    shifted = points - mins
    maxs = shifted.max(axis=0)
    num_boxes = np.ceil(maxs / box_size).astype(int) + 1
    grid = np.zeros(num_boxes)
    
    indices = np.floor(shifted / box_size).astype(int)
    for idx in indices: grid[tuple(idx)] += 1
    counts = grid.flatten()
    mean_val = counts.mean()
    var_val = counts.var()
    return var_val / (mean_val**2) + 1 if mean_val != 0 else np.nan

def analyze_component_structure(G_comp):
    num_loops = len(nx.cycle_basis(G_comp))
    abnormal_degree_nodes = [node for node, degree in G_comp.degree() if degree > 3]
    return num_loops, len(abnormal_degree_nodes)

def extract_segments_from_component_using_shortest_path(G_comp, distance_map):
    endpoints = [node for node, deg in G_comp.degree() if deg == 1]
    if not endpoints: return [[], [], []], [{}, {}, {}]

    diameters_eps = {node: 2 * distance_map.get(node, 0.5) for node in endpoints}
    sorted_eps = sorted(endpoints, key=lambda n: diameters_eps[n], reverse=True)

    root1 = sorted_eps[0]
    root2 = sorted_eps[1] if len(sorted_eps) > 1 else root1

    bif_nodes = [node for node, deg in G_comp.degree() if deg >= 3]
    if bif_nodes:
        diameters_bif = {node: 2 * distance_map.get(node, 0.5) for node in bif_nodes}
        root3 = max(bif_nodes, key=lambda n: diameters_bif[n])
    else:
        root3 = root1

    roots = [root1, root2, root3]
    segment_lists = []
    segment_counts_list = []

    for root in roots:
        segs = []
        counts = defaultdict(int)
        for ep in endpoints:
            if ep == root: continue
            # weight='weight' matters: without it this is a hop-count BFS, so
            # wherever a component contains a loop (and these graphs do -- the
            # code reports num_loops explicitly) it can return a route that is
            # not the physically shortest one, silently corrupting
            # geodesic_length, avg_diameter and every spline metric derived
            # from the path.
            try: path = nx.shortest_path(G_comp, source=root, target=ep, weight='weight')
            except nx.NetworkXNoPath: continue
            segs.append(path)
            for node in path: counts[node] += 1
        segment_lists.append(segs)
        segment_counts_list.append(dict(counts))

    return segment_lists, segment_counts_list

def save_results(results, output_folder, save_segment_masks, save_conn_comp_masks, region_true_volumes=None, patient_name=None):
    print("  Saving results...")
    file_prefix = f"{patient_name}_" if patient_name else ""
    general_keys = [
        'total_length', 'num_bifurcations', 'bifurcation_density', 'volume',
        'fractal_dimension', 'lacunarity', 'num_loops', 'num_abnormal_degree_nodes'
    ]

    all_rows = []
    region_rows = []

    region_results = results.get('region_metrics', {'global': results})
    if 'global' in region_results: print("  No regional metrics found, saving global results only.")

    for region_label, region_data in region_results.items():
        if not isinstance(region_data, dict): continue 
        components = {k: v for k, v in region_data.items() if isinstance(v, dict) and 'total_length' in v}
        sorted_items = sorted(components.items(), key=lambda item: item[1].get('total_length', 0), reverse=True)

        for new_idx, (cid, data) in enumerate(sorted_items):
            comp_idx = new_idx + 1
            region_name = f"Region_{region_label}"
            comp_dir = os.path.join(output_folder, "regional_results" ,region_name, f"Conn_comp_{comp_idx}")
            Path(comp_dir).mkdir(parents=True, exist_ok=True)

            row = {'region_label': region_label, 'component_index': comp_idx, 'original_component_id': cid}
            for k in general_keys: row[k] = data.get(k, np.nan)

            agg = data.get('aggregated_tortuosity_by_root', [])
            prefixes = ['Largest_endpoint_root_', '2nd_Largest_endpoint_root_', 'Largest_bifurcation_root_']
            for i, prefix in enumerate(prefixes):
                if i < len(agg):
                    row[prefix + 'mean_curvature'] = agg[i].get('mean_curvature', np.nan)
                    row[prefix + 'mean_square_curvature'] = agg[i].get('mean_square_curvature', np.nan)
                else:
                    row[prefix + 'mean_curvature'] = np.nan
                    row[prefix + 'mean_square_curvature'] = np.nan

            all_rows.append({k: v for k, v in row.items() if k != 'region_label'})
            region_rows.append(row)

            if save_conn_comp_masks and 'reconstructed_conn_comp' in data:
                nib.save(data['reconstructed_conn_comp'], os.path.join(comp_dir, f'Conn_comp_{comp_idx}_skeleton.nii.gz'))

            if 'segments_by_root' in data:
                segs_dir = os.path.join(comp_dir, 'Segments')
                os.makedirs(segs_dir, exist_ok=True)
                root_names = ['Largest endpoint root', 'Second largest endpoint root', 'Largest bifurcation root']
                
                for root_idx, root_entry in enumerate(data['segments_by_root']):
                    if root_idx >= len(root_names): break
                    root_dir = os.path.join(segs_dir, root_names[root_idx])
                    os.makedirs(root_dir, exist_ok=True)

                    metrics_list = root_entry.get('segment_metrics', [])
                    for seg_idx, sm in enumerate(metrics_list, start=1):
                        seg_dir = os.path.join(root_dir, f"Segment_{seg_idx}")
                        os.makedirs(seg_dir, exist_ok=True)
                        pd.DataFrame([sm]).to_csv(os.path.join(seg_dir, 'Segment_metrics.csv'), index=False)

    if region_rows:
        df_detailed = pd.DataFrame(region_rows)
        df_detailed = df_detailed.drop(columns=['component_index', 'original_component_id'], errors='ignore')
        df_detailed = df_detailed.sort_values(by=['region_label', 'total_length'], ascending=[True, False])
        df_detailed.to_csv(os.path.join(output_folder, f'{file_prefix}all_components_by_region.csv'), index=False)

        summary_metrics = ['total_length', 'num_bifurcations', 'volume', 'num_loops', 'num_abnormal_degree_nodes']
        tortuosity_cols = [
            'Largest_endpoint_root_mean_curvature', 'Largest_endpoint_root_mean_square_curvature',
            '2nd_Largest_endpoint_root_mean_curvature', '2nd_Largest_endpoint_root_mean_square_curvature',
            'Largest_bifurcation_root_mean_curvature', 'Largest_bifurcation_root_mean_square_curvature'
        ]

        region_groups = df_detailed.groupby('region_label')
        summary_rows = []

        for region_label, group in region_groups:
            row = {'region_label': region_label, 'num_components': len(group)}
            for metric in summary_metrics: row[metric] = group[metric].sum()

            # Apply the ground-truth mask volume once, at the aggregate region
            # level, instead of on a single component row -- doing it earlier
            # (per-component) would get re-summed here alongside sibling
            # components' cylinder approximations and double-count it.
            if region_true_volumes and region_label in region_true_volumes:
                row['volume_approx_cylinder'] = row.get('volume', np.nan)
                row['volume'] = region_true_volumes[region_label]

            total_len = group['total_length'].sum()
            for col in tortuosity_cols:
                if col in group:
                    # Exclude undefined values rather than zeroing them: a
                    # component with no measurable curvature must drop out of
                    # BOTH numerator and denominator. fillna(0) kept its full
                    # length in the denominator while contributing nothing,
                    # dragging the regional average toward zero.
                    valid = group[col].notna()
                    w = group['total_length'][valid]
                    w_sum = w.sum()
                    row[col] = ((group[col][valid] * w).sum() / w_sum) if w_sum > 0 else np.nan
                else:
                    row[col] = np.nan

            region_data = region_results.get(region_label, {})
            for b in ['betti_0', 'betti_1', 'betti_2']: row[b] = region_data.get(b, np.nan)
            summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows)
        df_summary = df_summary.sort_values(by='region_label')
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


def compute_metrics_for_mask(G, node_radius_map, selected_metrics):
    results = {}
    ensure_edge_weights(G)
    for cid, comp_nodes in enumerate(nx.connected_components(G)):
        Gc = G.subgraph(comp_nodes).copy()
        data = {}

        coords = np.array([Gc.nodes[n]['pos'] for n in Gc.nodes()])
        total_len = Gc.size(weight='weight')
        degrees = dict(Gc.degree())
        num_bif = sum(1 for d in degrees.values() if d >= 3)
        
        if 'num_bifurcations' in selected_metrics: data['num_bifurcations'] = num_bif
        if 'total_length' in selected_metrics: data['total_length'] = total_len
        if 'bifurcation_density' in selected_metrics: data['bifurcation_density'] = num_bif / total_len if total_len > 0 else np.nan
        
        if 'volume' in selected_metrics:
            # Conical frustum, not a mean-radius cylinder. The two differ by
            # pi*l*(r_u - r_v)**2 / 12, which is always >= 0, so the old form
            # systematically under-estimated volume wherever the radius
            # changes along an edge -- i.e. at every taper and bifurcation.
            vol = 0.0
            for u, v, d in Gc.edges(data=True):
                l = d.get('weight', 0)
                r_u = node_radius_map.get(u, 0.5)
                r_v = node_radius_map.get(v, 0.5)
                vol += np.pi * l * (r_u ** 2 + r_u * r_v + r_v ** 2) / 3.0
            data['volume'] = vol
        
        if 'fractal_dimension' in selected_metrics: data['fractal_dimension'] = compute_fractal_dimension(coords)
        if 'lacunarity' in selected_metrics:
            box_size = np.max(coords.max(axis=0) - coords.min(axis=0)) / 10 or 1
            data['lacunarity'] = calculate_lacunarity(coords, box_size)
        
        if 'num_loops' in selected_metrics or 'num_abnormal_degree_nodes' in selected_metrics:
            nl, nab = analyze_component_structure(Gc)
            if 'num_loops' in selected_metrics: data['num_loops'] = nl
            if 'num_abnormal_degree_nodes' in selected_metrics: data['num_abnormal_degree_nodes'] = nab

        if any(m in selected_metrics for m in ['geodesic_length', 'avg_diameter', 'spline_mean_curvature', 'spline_mean_square_curvature']):
            seg_lists, seg_counts = extract_segments_from_component_using_shortest_path(Gc, node_radius_map)
            segments_info = []
            agg_curv = []
            agg_curv2 = []
            
            for r_idx, seg_list in enumerate(seg_lists):
                seg_metrics = []
                # Separate weight accumulator per metric. A segment with fewer
                # than 4 points yields NaN curvature (routine for short
                # terminal branches), and adding its length to a shared
                # denominator while contributing nothing to the numerator
                # biases the length-weighted average toward zero. Each metric
                # therefore only accumulates the length of the segments that
                # actually defined it.
                len_curv = 0.0
                len_curv2 = 0.0
                sum_curv = 0.0
                sum_curv2 = 0.0

                for seg in seg_list:
                    pts = np.array([Gc.nodes[n]['pos'] for n in seg])
                    sm = {}
                    L_geo = sum(np.linalg.norm(pts[i] - pts[i+1]) for i in range(len(pts)-1))
                    sm['geodesic_length'] = L_geo

                    if 'avg_diameter' in selected_metrics: sm['avg_diameter'] = np.mean([2 * node_radius_map.get(n, 0.5) for n in seg])
                        
                    counts_arr = np.array([seg_counts[r_idx].get(n, 1) for n in seg])
                    tort = compute_tortuosity_metrics(pts, smoothing=0, n_samples=500, counts=counts_arr)
                    sm.update(tort)

                    if 'spline_mean_curvature' in selected_metrics:
                        sm['spline_mean_curvature'] = tort['spline_mean_curvature']
                        if not np.isnan(tort['spline_mean_curvature']):
                            sum_curv += tort['spline_mean_curvature'] * L_geo
                            len_curv += L_geo

                    if 'spline_mean_square_curvature' in selected_metrics:
                        sm['spline_mean_square_curvature'] = tort['spline_mean_square_curvature']
                        if not np.isnan(tort['spline_mean_square_curvature']):
                            sum_curv2 += tort['spline_mean_square_curvature'] * L_geo
                            len_curv2 += L_geo

                    seg_metrics.append(sm)

                agg_curv.append(sum_curv / len_curv if len_curv > 0 else np.nan)
                agg_curv2.append(sum_curv2 / len_curv2 if len_curv2 > 0 else np.nan)

                segments_info.append({'segment_metrics': seg_metrics})

            data['segments_by_root'] = segments_info
            data['aggregated_tortuosity_by_root'] = [
                {'mean_curvature': agg_curv[i], 'mean_square_curvature': agg_curv2[i]} for i in range(len(agg_curv))
            ]

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

    import sys
    import networkx as nx
    import numpy as np

    # Legacy numpy compatibility (kept from original code)
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
        
        # Rebuild the graph from the node list
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

    # Fallback: Build radius map from node attributes if not provided separately
    if node_radius_map is None:
        node_radius_map = {}
        for n in G.nodes():
            node_radius_map[n] = G.nodes[n].get('radius', 0.5) 

    return G, node_radius_map

def extract_metrics(output_folder, graph_pkl_path, selected_metrics, label_map_path=None, mask_path=None, registered_atlas_path=None):
    use_atlas = True if registered_atlas_path else False
    results = {'region_metrics': {}} if use_atlas else {}
    
    # Auto-extract patient name from the graph filename
    patient_name = os.path.basename(graph_pkl_path)
    if patient_name.endswith('_vessel_data.pkl'):
        patient_name = patient_name[: -len('_vessel_data.pkl')]
    else:
        patient_name = os.path.splitext(patient_name)[0]
    
    if use_atlas: aligned_atlas = sitk.ReadImage(registered_atlas_path)
    
    try:
        G, node_radius_map = robust_graph_loader(graph_pkl_path)
    except Exception as e:
        print(f"  ERROR: Failed to load pickle - {e}")
        return
    
    # --- LOAD TRUE MASK FOR EXACT VOLUME ---
    true_mask_data = None
    voxel_volume_mm3 = 0.0
    total_true_volume = np.nan  # NEW: Initialize global volume
    
    if mask_path and os.path.exists(mask_path):
        try:
            mask_img = nib.load(mask_path)
            true_mask_data = np.asarray(mask_img.dataobj) > 0
            voxel_volume_mm3 = np.prod(mask_img.header.get_zooms()[:3])
            
            # NEW: Calculate the absolute total volume of the entire mask
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
        
        region_nodes = {}
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
                    
                    # FILTER: Keep only the 30 arterial territories, ignore ventricles (31, 32)
                    if 0 < region_id <= 30:
                        if region_id not in region_nodes: region_nodes[region_id] = []
                        region_nodes[region_id].append(n)
            except Exception as e:
                raise ValueError(f"Failed to transform point {pos} to index in atlas. Error: {e}")
            
        region_true_volumes = {}
        for r_id in sorted(region_nodes.keys()):
            r_id_int = int(r_id) # Absolute safety cast for dictionary matching
            r_name = label_map.get(r_id_int, f"region_{r_id_int}")

            sub_G = G.subgraph(region_nodes[r_id]).copy()
            region_result = compute_metrics_for_mask(sub_G, node_radius_map, selected_metrics)

            # Record True Regional Volume (applied once at region-summary level
            # in save_results, not per-component -- see region_true_volumes).
            if 'volume' in selected_metrics and true_mask_data is not None:
                region_voxel_count = np.sum(true_mask_data & (aligned_atlas_data == r_id))
                region_true_volumes[r_id_int] = region_voxel_count * voxel_volume_mm3

            results['region_metrics'][r_id_int] = region_result

        save_results(results, output_folder, save_segment_masks=None, save_conn_comp_masks=None, region_true_volumes=region_true_volumes, patient_name=patient_name)

        n_regions = len(region_nodes)
        print(f"  Found {n_regions} arterial regions. Generating regional VTPs for {patient_name}...")
        regional_actors = []
        regions_vtp_dir = os.path.join(output_folder, "regional_vtps")
        os.makedirs(regions_vtp_dir, exist_ok=True)
        
        for r_id, nodes in region_nodes.items():
            r_id_int = int(r_id)
            sub_G = G.subgraph(nodes)
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
            
        # --- NEW: SAVE TRUE GLOBAL TOTALS FOR ALL METRICS ---
        print("  Computing true global metrics on the full unparcellated graph...")
        # Compute all metrics on the full graph G
        true_global_result = compute_metrics_for_mask(G, node_radius_map, selected_metrics)
        
        # Aggregate all disconnected components for the global summary
        global_data = {
            'total_length': 0.0,
            'num_bifurcations': 0,
            'volume': 0.0,
            'num_loops': 0,
            'num_abnormal_degree_nodes': 0
        }
        
        # Variables for length-weighted tortuosity averages
        tortuosity_cols = [
            'Largest_endpoint_root_mean_curvature', 'Largest_endpoint_root_mean_square_curvature',
            '2nd_Largest_endpoint_root_mean_curvature', '2nd_Largest_endpoint_root_mean_square_curvature',
            'Largest_bifurcation_root_mean_curvature', 'Largest_bifurcation_root_mean_square_curvature'
        ]
        tortuosity_sums = {col: 0.0 for col in tortuosity_cols}
        # Per-column length accumulator: only components that actually defined
        # a given metric may contribute their length to its denominator (see
        # the same fix in compute_metrics_for_mask and save_results).
        tortuosity_lens = {col: 0.0 for col in tortuosity_cols}
        
        for cid, comp_data in true_global_result.items():
            comp_length = comp_data.get('total_length', 0)
            
            # Sum up additive metrics
            global_data['total_length'] += comp_length
            global_data['num_bifurcations'] += comp_data.get('num_bifurcations', 0)
            global_data['volume'] += comp_data.get('volume', 0)
            global_data['num_loops'] += comp_data.get('num_loops', 0)
            global_data['num_abnormal_degree_nodes'] += comp_data.get('num_abnormal_degree_nodes', 0)
            
            # Accumulate length-weighted tortuosity
            agg = comp_data.get('aggregated_tortuosity_by_root', [])
            prefixes = ['Largest_endpoint_root_', '2nd_Largest_endpoint_root_', 'Largest_bifurcation_root_']
            
            for i, prefix in enumerate(prefixes):
                if i < len(agg):
                    mean_curv = agg[i].get('mean_curvature', np.nan)
                    mean_sq_curv = agg[i].get('mean_square_curvature', np.nan)
                    
                    if not np.isnan(mean_curv):
                        tortuosity_sums[prefix + 'mean_curvature'] += mean_curv * comp_length
                        tortuosity_lens[prefix + 'mean_curvature'] += comp_length
                    if not np.isnan(mean_sq_curv):
                        tortuosity_sums[prefix + 'mean_square_curvature'] += mean_sq_curv * comp_length
                        tortuosity_lens[prefix + 'mean_square_curvature'] += comp_length

        # Finalize length-weighted averages, each over its own valid length
        for col in tortuosity_cols:
            w = tortuosity_lens[col]
            global_data[col] = (tortuosity_sums[col] / w) if w > 0 else np.nan
                
        # Inject True Global Volume from mask if available
        if 'volume' in selected_metrics and true_mask_data is not None:
            global_data['volume_approx_cylinder'] = global_data.get('volume', 0)
            global_data['volume'] = total_true_volume
            
        global_df = pd.DataFrame([global_data])
        global_df.to_csv(os.path.join(output_folder, f'{patient_name}_global_summary.csv'), index=False)
    
    else:
        print("  Computing global vessel metrics...")
        parent, child = os.path.split(output_folder.rstrip('/'))
        output_folder = os.path.join(parent + "_GLOBAL", child)
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        global_result = compute_metrics_for_mask(G, node_radius_map, selected_metrics)

        # Record True Global Volume under the 'global' pseudo-region key that
        # save_results() uses when there's no atlas (applied once at the
        # region-summary level, not per-component).
        region_true_volumes = {}
        if 'volume' in selected_metrics and true_mask_data is not None:
            total_voxel_count = np.sum(true_mask_data)
            region_true_volumes['global'] = total_voxel_count * voxel_volume_mm3

        results.update(global_result)
        save_results(results, output_folder, save_segment_masks=None, save_conn_comp_masks=None, region_true_volumes=region_true_volumes, patient_name=patient_name)

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

    parser.add_argument("--metrics", nargs='+', default=[
        'total_length', 'num_bifurcations', 'bifurcation_density', 'volume',
        'fractal_dimension', 'lacunarity', 'geodesic_length', 'avg_diameter',
        'spline_arc_length', 'spline_chord_length', 'spline_mean_curvature',
        'spline_mean_square_curvature', 'spline_rms_curvature', 'arc_over_chord',
        'num_loops', 'num_abnormal_degree_nodes'
    ], help="List of metrics to compute.")

    args = parser.parse_args()

    extract_metrics(
        output_folder=args.out_dir,
        graph_pkl_path=args.graph,
        selected_metrics=args.metrics,
        label_map_path=args.label_map,
        mask_path=args.mask,
        registered_atlas_path=args.atlas
    )