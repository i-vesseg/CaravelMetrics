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
    if pts.shape[0] < 4:
        nan_dict = {k: np.nan for k in [
            'spline_arc_length','spline_chord_length',
            'spline_mean_curvature','spline_mean_square_curvature',
            'spline_rms_curvature','arc_over_chord','fit_rmse']}
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
            try: path = nx.shortest_path(G_comp, source=root, target=ep)
            except nx.NetworkXNoPath: continue
            segs.append(path)
            for node in path: counts[node] += 1
        segment_lists.append(segs)
        segment_counts_list.append(dict(counts))

    return segment_lists, segment_counts_list

def save_results(results, output_folder, save_segment_masks, save_conn_comp_masks):
    print("  Saving results...")
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
        df_detailed.to_csv(os.path.join(output_folder, 'all_components_by_region.csv'), index=False)

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

            total_len = group['total_length'].sum()
            for col in tortuosity_cols:
                if col in group:
                    values = group[col].fillna(0)
                    weighted_avg = (values * group['total_length']).sum() / total_len if total_len > 0 else np.nan
                    row[col] = weighted_avg
                else:
                    row[col] = np.nan

            region_data = region_results.get(region_label, {})
            for b in ['betti_0', 'betti_1', 'betti_2']: row[b] = region_data.get(b, np.nan)
            summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows)
        df_summary = df_summary.sort_values(by='region_label')
        df_summary.to_csv(os.path.join(output_folder, 'region_summary.csv'), index=False)

    print(f"Results saved to: {output_folder}")

    
def compute_metrics_for_mask(G, node_radius_map, selected_metrics):
    results = {}
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
            vol = 0.0    
            for u, v, d in Gc.edges(data=True):
                l = d.get('weight', 0)
                r = (node_radius_map.get(u, 0.5) + node_radius_map.get(v, 0.5)) / 2.0
                vol += l * np.pi * (r ** 2) 
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
                total_geo = 0.0
                sum_curv = 0.0
                sum_curv2 = 0.0

                for seg in seg_list:
                    pts = np.array([Gc.nodes[n]['pos'] for n in seg])
                    sm = {}
                    L_geo = sum(np.linalg.norm(pts[i] - pts[i+1]) for i in range(len(pts)-1))
                    sm['geodesic_length'] = L_geo
                    total_geo += L_geo

                    if 'avg_diameter' in selected_metrics: sm['avg_diameter'] = np.mean([2 * node_radius_map.get(n, 0.5) for n in seg])
                        
                    counts_arr = np.array([seg_counts[r_idx].get(n, 1) for n in seg])
                    tort = compute_tortuosity_metrics(pts, smoothing=0, n_samples=500, counts=counts_arr)
                    sm.update(tort)

                    if 'spline_mean_curvature' in selected_metrics:
                        sm['spline_mean_curvature'] = tort['spline_mean_curvature']
                        if not np.isnan(tort['spline_mean_curvature']): sum_curv += tort['spline_mean_curvature'] * L_geo
                            
                    if 'spline_mean_square_curvature' in selected_metrics:
                        sm['spline_mean_square_curvature'] = tort['spline_mean_square_curvature']
                        if not np.isnan(tort['spline_mean_square_curvature']): sum_curv2 += tort['spline_mean_square_curvature'] * L_geo
                            
                    seg_metrics.append(sm)

                if total_geo > 0:
                    agg_curv.append(sum_curv/total_geo)
                    agg_curv2.append(sum_curv2/total_geo)
                else:
                    agg_curv.append(np.nan)
                    agg_curv2.append(np.nan)

                segments_info.append({'segment_metrics': seg_metrics})

            data['segments_by_root'] = segments_info
            data['aggregated_tortuosity_by_root'] = [
                {'mean_curvature': agg_curv[i], 'mean_square_curvature': agg_curv2[i]} for i in range(len(agg_curv))
            ]

        results[cid] = data
    return results


def extract_metrics(output_folder, graph_pkl_path, selected_metrics, label_map_path=None, mask_path=None, registered_atlas_path=None):
    use_atlas = True if registered_atlas_path else False
    results = {'region_metrics': {}} if use_atlas else {}
    
    # Auto-extract patient name from the graph filename
    patient_name = os.path.basename(graph_pkl_path).replace('_vessel_data.pkl', '').replace('.pkl', '')
    
    if use_atlas: aligned_atlas = sitk.ReadImage(registered_atlas_path)
    
    try:
        import sys
        if 'numpy._core' not in sys.modules:
            try:
                import numpy.core as _core
                sys.modules['numpy._core'] = _core
            except: pass
        with open(graph_pkl_path, 'rb') as f:
            try: data = pickle.load(f, encoding='latin1')
            except:
                f.seek(0)
                data = pickle.load(f)
        G = data['graph']
        node_radius_map = data['node_radius_map']
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
            true_mask_data = mask_img.get_fdata() > 0
            voxel_volume_mm3 = np.prod(mask_img.header.get_zooms()[:3])
            
            # NEW: Calculate the absolute total volume of the entire mask
            total_true_volume = np.sum(true_mask_data) * voxel_volume_mm3 
            
            print(f"  Loaded True Mask for exact volume computation (Voxel size: {voxel_volume_mm3:.4f} mm³)")
        except Exception as e:
            print(f"  Warning: Could not load true mask: {e}")
    
    label_map = load_label_map(label_map_path)
    
    if use_atlas:
        # Snap atlas data to nearest integers to fix float inaccuracies
        aligned_atlas_data = np.round(sitk.GetArrayFromImage(aligned_atlas)).astype(int).T
        region_nodes = {}
        size = aligned_atlas.GetSize()
        for n in G.nodes():
            pos = G.nodes[n]['pos']
            try:
                idx = aligned_atlas.TransformPhysicalPointToIndex(pos)
                if (0 <= idx[0] < size[0] and 0 <= idx[1] < size[1] and 0 <= idx[2] < size[2]):
                    raw_val = aligned_atlas[idx]
                    region_id = int(round(float(raw_val))) # Force standard integer
                    
                    # FILTER: Keep only the 30 arterial territories, ignore ventricles (31, 32)
                    if 0 < region_id <= 30:
                        if region_id not in region_nodes: region_nodes[region_id] = []
                        region_nodes[region_id].append(n)
            except:
                raise ValueError(f"Failed to transform point {pos} to index in atlas.")
            
        for r_id in sorted(region_nodes.keys()):
            r_id_int = int(r_id) # Absolute safety cast for dictionary matching
            r_name = label_map.get(r_id_int, f"region_{r_id_int}")
            
            sub_G = G.subgraph(region_nodes[r_id]).copy()
            region_result = compute_metrics_for_mask(sub_G, node_radius_map, selected_metrics)
            
            # Inject True Regional Volume
            if 'volume' in selected_metrics and true_mask_data is not None:
                region_voxel_count = np.sum(true_mask_data & (aligned_atlas_data == r_id))
                true_vol = region_voxel_count * voxel_volume_mm3
                if 0 in region_result:
                    region_result[0]['volume_approx_cylinder'] = region_result[0].get('volume', 0)
                    region_result[0]['volume'] = true_vol
                    
            results['region_metrics'][r_id_int] = region_result
        
        save_results(results, output_folder, save_segment_masks=None, save_conn_comp_masks=None)
    
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
            
        # --- NEW: SAVE GLOBAL TOTALS ---
        global_length = G.size(weight='weight')
        global_df = pd.DataFrame([{
            'total_global_volume_mm3': total_true_volume,
            'total_global_length_mm': global_length
        }])
        global_df.to_csv(os.path.join(output_folder, 'global_summary.csv'), index=False)
    
    else:
        print("  Computing global vessel metrics...")
        parent, child = os.path.split(output_folder.rstrip('/'))
        output_folder = os.path.join(parent + "_GLOBAL", child)
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        global_result = compute_metrics_for_mask(G, node_radius_map, selected_metrics)
        
        # Inject True Global Volume
        if 'volume' in selected_metrics and true_mask_data is not None:
            total_voxel_count = np.sum(true_mask_data)
            true_vol = total_voxel_count * voxel_volume_mm3
            if 0 in global_result:
                global_result[0]['volume_approx_cylinder'] = global_result[0].get('volume', 0)
                global_result[0]['volume'] = true_vol
                
        results.update(global_result)
        save_results(results, output_folder, save_segment_masks=None, save_conn_comp_masks=None)

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
        'fit_rmse', 'num_loops', 'num_abnormal_degree_nodes'
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