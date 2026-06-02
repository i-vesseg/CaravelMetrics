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
import json
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
import vtk
from pathlib import Path


ROOT_CRITERIA = [
    {
        'id': 1,
        'key': 'largest_endpoint',
        'name': 'Largest endpoint root',
        'prefix': 'Largest_endpoint_root_',
    },
    {
        'id': 2,
        'key': 'second_largest_endpoint',
        'name': 'Second largest endpoint root',
        'prefix': '2nd_Largest_endpoint_root_',
    },
    {
        'id': 3,
        'key': 'largest_bifurcation',
        'name': 'Largest bifurcation root',
        'prefix': 'Largest_bifurcation_root_',
    },
]

SEGMENT_METRIC_KEYS = [
    'geodesic_length',
    'avg_diameter',
    'spline_arc_length',
    'spline_chord_length',
    'spline_mean_curvature',
    'spline_mean_square_curvature',
    'spline_rms_curvature',
    'arc_over_chord',
    'fit_rmse',
]


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


def _build_root_info(criterion, root, distance_map, fallback=False):
    diameter = None if root is None else 2 * distance_map.get(root, 0.5)
    return {
        'root_criterion_id': criterion['id'],
        'root_criterion_key': criterion['key'],
        'root_criterion_name': criterion['name'],
        'root_node': root,
        'root_diameter': diameter,
        'fallback_root': fallback,
    }


def extract_segments_from_component_using_shortest_path(G_comp, distance_map):
    """
    From the connected component G_comp, identify all endpoints (degree == 1)
    and select all three current root criteria. It returns root-to-endpoint
    shortest-path segments, path-multiplicity counts, and root metadata.
    """
    endpoints = [node for node, deg in G_comp.degree() if deg == 1]
    if not endpoints:
        root_infos = [_build_root_info(c, None, distance_map) for c in ROOT_CRITERIA]
        return [[], [], []], [{}, {}, {}], root_infos

    diameters_eps = {node: 2 * distance_map.get(node, 0.5) for node in endpoints}
    sorted_eps = sorted(endpoints, key=lambda n: diameters_eps[n], reverse=True)

    root1 = sorted_eps[0]
    root2 = sorted_eps[1] if len(sorted_eps) > 1 else root1

    bif_nodes = [node for node, deg in G_comp.degree() if deg >= 3]
    if bif_nodes:
        diameters_bif = {node: 2 * distance_map.get(node, 0.5) for node in bif_nodes}
        root3 = max(bif_nodes, key=lambda n: diameters_bif[n])
        root3_is_fallback = False
    else:
        root3 = root1
        root3_is_fallback = True

    roots = [root1, root2, root3]
    root_infos = [
        _build_root_info(ROOT_CRITERIA[0], root1, distance_map),
        _build_root_info(ROOT_CRITERIA[1], root2, distance_map, fallback=(root2 == root1 and len(sorted_eps) == 1)),
        _build_root_info(ROOT_CRITERIA[2], root3, distance_map, fallback=root3_is_fallback),
    ]
    segment_lists = []
    segment_counts_list = []

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
            for node in path:
                counts[node] += 1
        segment_lists.append(segs)
        segment_counts_list.append(dict(counts))

    return segment_lists, segment_counts_list, root_infos


def _to_python_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return _to_python_scalar(value)


def _node_to_json(node):
    return _json_safe(node)


def _node_to_csv(node):
    if isinstance(node, (list, tuple, np.ndarray)):
        return json.dumps(_json_safe(node))
    return _to_python_scalar(node)


def _node_to_vtp_id(node):
    try:
        return int(node)
    except (TypeError, ValueError):
        return -1


def _region_to_vtp_id(region_label):
    try:
        return int(region_label)
    except (TypeError, ValueError):
        return -1


def _metric_value(metrics, key):
    return _to_python_scalar(metrics.get(key)) if key in metrics else None


def _format_node_path(node_path):
    return ';'.join(str(_node_to_csv(n)) for n in node_path)


def _build_segment_definition(segment_uid, region_label, component_index, original_component_id, root_entry, segment_entry, metrics):
    node_path = segment_entry.get('node_path', [])
    point_path = segment_entry.get('point_path', [])
    edge_path = [[node_path[i], node_path[i + 1]] for i in range(max(0, len(node_path) - 1))]
    return {
        'segment_uid': segment_uid,
        'region_label': _json_safe(region_label),
        'component_index': component_index,
        'original_component_id': _json_safe(original_component_id),
        'root_criterion_id': root_entry.get('root_criterion_id'),
        'root_criterion_key': root_entry.get('root_criterion_key'),
        'root_criterion_name': root_entry.get('root_criterion_name'),
        'segment_index': segment_entry.get('segment_index'),
        'root_node': _node_to_json(segment_entry.get('root_node')),
        'endpoint_node': _node_to_json(segment_entry.get('endpoint_node')),
        'root_diameter': _json_safe(root_entry.get('root_diameter')),
        'fallback_root': bool(root_entry.get('fallback_root', False)),
        'node_path': _json_safe(node_path),
        'edge_path': _json_safe(edge_path),
        'point_path': _json_safe(point_path),
        'node_count': len(node_path),
        'edge_count': max(0, len(node_path) - 1),
        'point_count': len(point_path),
        'metrics': _json_safe(metrics),
    }


def _segment_definition_csv_rows(segment_definitions):
    rows = []
    for definition in segment_definitions:
        metrics = definition.get('metrics', {})
        row = {
            'segment_uid': definition.get('segment_uid'),
            'region_label': definition.get('region_label'),
            'component_index': definition.get('component_index'),
            'original_component_id': definition.get('original_component_id'),
            'root_criterion_id': definition.get('root_criterion_id'),
            'root_criterion_key': definition.get('root_criterion_key'),
            'root_criterion_name': definition.get('root_criterion_name'),
            'segment_index': definition.get('segment_index'),
            'root_node': _node_to_csv(definition.get('root_node')),
            'endpoint_node': _node_to_csv(definition.get('endpoint_node')),
            'root_diameter': definition.get('root_diameter'),
            'fallback_root': definition.get('fallback_root'),
            'node_count': definition.get('node_count'),
            'edge_count': definition.get('edge_count'),
            'point_count': definition.get('point_count'),
            'node_path': _format_node_path(definition.get('node_path', [])),
        }
        for key in SEGMENT_METRIC_KEYS:
            row[key] = _metric_value(metrics, key)
        rows.append(row)
    return rows


def _write_segment_vtp(segment_definitions, output_path):
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    array_names = [
        'SegmentUID', 'SegmentIndex', 'RootCriterionID', 'RegionLabel',
        'ComponentIndex', 'OriginalComponentID', 'RootNodeID', 'EndpointNodeID',
    ]
    arrays = {}
    for name in array_names:
        arr = vtk.vtkIntArray()
        arr.SetName(name)
        arrays[name] = arr

    for definition in segment_definitions:
        point_path = definition.get('point_path', [])
        if len(point_path) < 2:
            continue
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(point_path))
        for i, point in enumerate(point_path):
            point_id = points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
            polyline.GetPointIds().SetId(i, point_id)
        lines.InsertNextCell(polyline)
        arrays['SegmentUID'].InsertNextValue(int(definition.get('segment_uid', -1)))
        arrays['SegmentIndex'].InsertNextValue(int(definition.get('segment_index', -1)))
        arrays['RootCriterionID'].InsertNextValue(int(definition.get('root_criterion_id', -1)))
        arrays['RegionLabel'].InsertNextValue(_region_to_vtp_id(definition.get('region_label')))
        arrays['ComponentIndex'].InsertNextValue(int(definition.get('component_index', -1)))
        arrays['OriginalComponentID'].InsertNextValue(_node_to_vtp_id(definition.get('original_component_id')))
        arrays['RootNodeID'].InsertNextValue(_node_to_vtp_id(definition.get('root_node')))
        arrays['EndpointNodeID'].InsertNextValue(_node_to_vtp_id(definition.get('endpoint_node')))

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetLines(lines)
    for arr in arrays.values():
        poly_data.GetCellData().AddArray(arr)
    poly_data.GetCellData().SetActiveScalars('SegmentUID')

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(poly_data)
    writer.Write()


def _write_segment_definition_outputs(output_folder, segment_definitions):
    package = {
        'schema_version': 1,
        'root_criteria': ROOT_CRITERIA,
        'segments': segment_definitions,
    }
    json_path = os.path.join(output_folder, 'segment_definitions.json')
    csv_path = os.path.join(output_folder, 'segment_definitions.csv')
    pkl_path = os.path.join(output_folder, 'segment_definitions.pkl')
    vtp_path = os.path.join(output_folder, 'vessel_segments.vtp')

    with open(json_path, 'w') as f:
        json.dump(_json_safe(package), f, indent=2, allow_nan=False)

    csv_columns = [
        'segment_uid', 'region_label', 'component_index', 'original_component_id',
        'root_criterion_id', 'root_criterion_key', 'root_criterion_name',
        'segment_index', 'root_node', 'endpoint_node', 'root_diameter',
        'fallback_root', 'node_count', 'edge_count', 'point_count', 'node_path',
    ] + SEGMENT_METRIC_KEYS
    pd.DataFrame(_segment_definition_csv_rows(segment_definitions), columns=csv_columns).to_csv(csv_path, index=False)

    with open(pkl_path, 'wb') as f:
        pickle.dump(package, f)

    _write_segment_vtp(segment_definitions, vtp_path)
    print(f"  Saved segment definitions: {json_path}")
    print(f"  Saved segment definition index: {csv_path}")
    print(f"  Saved segment definition PKL: {pkl_path}")
    print(f"  Saved segment VTP: {vtp_path}")

def save_results(results, output_folder, save_segment_masks, save_conn_comp_masks):

    print("  Saving results...")
    general_keys = [
        'total_length', 'num_bifurcations', 'bifurcation_density', 'volume',
        'fractal_dimension', 'lacunarity',
        'num_loops', 'num_abnormal_degree_nodes'
    ]

    all_rows = []
    region_rows = []
    segment_definitions = []
    segment_uid = 1

    if 'region_metrics' in results:
        region_results = results['region_metrics']
    else:
        print("  No regional metrics found, saving global results only.")
        region_results = {'global': results}

    for region_label, region_data in region_results.items():
        if not isinstance(region_data, dict):
            continue
        components = {
            k: v for k, v in region_data.items()
            if isinstance(v, dict) and ('total_length' in v or 'segments_by_root' in v)
        }
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

            agg = data.get('aggregated_tortuosity_by_root', [])
            for i, criterion in enumerate(ROOT_CRITERIA):
                prefix = criterion['prefix']
                if i < len(agg):
                    row[prefix + 'mean_curvature'] = agg[i].get('mean_curvature', np.nan)
                    row[prefix + 'mean_square_curvature'] = agg[i].get('mean_square_curvature', np.nan)
                else:
                    row[prefix + 'mean_curvature'] = np.nan
                    row[prefix + 'mean_square_curvature'] = np.nan

            all_rows.append({k: v for k, v in row.items() if k != 'region_label'})
            region_rows.append(row)

            if save_conn_comp_masks:
                nib.save(
                    data['reconstructed_conn_comp'],
                    os.path.join(comp_dir, f'Conn_comp_{comp_idx}_skeleton.nii.gz')
                )

            if 'segments_by_root' in data:
                segs_dir = os.path.join(comp_dir, 'Segments')
                os.makedirs(segs_dir, exist_ok=True)
                for root_idx, root_entry in enumerate(data['segments_by_root']):
                    if root_idx >= len(ROOT_CRITERIA):
                        break
                    root_dir = os.path.join(segs_dir, root_entry.get('root_criterion_name', ROOT_CRITERIA[root_idx]['name']))
                    os.makedirs(root_dir, exist_ok=True)

                    metrics_list = root_entry.get('segment_metrics', [])
                    segment_entries = root_entry.get('segments', [])
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

                    for segment_entry in segment_entries:
                        seg_idx = segment_entry.get('segment_index', 0)
                        metrics = metrics_list[seg_idx - 1] if 0 < seg_idx <= len(metrics_list) else {}
                        segment_definitions.append(_build_segment_definition(
                            segment_uid, region_label, comp_idx, cid, root_entry, segment_entry, metrics
                        ))
                        segment_uid += 1

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

        summary_metrics = [
            'total_length', 'num_bifurcations', 'volume', 'num_loops', 'num_abnormal_degree_nodes'
        ]
        tortuosity_cols = []
        for criterion in ROOT_CRITERIA:
            tortuosity_cols.extend([
                criterion['prefix'] + 'mean_curvature',
                criterion['prefix'] + 'mean_square_curvature',
            ])

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

    _write_segment_definition_outputs(output_folder, segment_definitions)
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

        seg_lists, seg_counts, root_infos = extract_segments_from_component_using_shortest_path(Gc, node_radius_map)
        segments_info = []
        agg_curv = []
        agg_curv2 = []
        compute_segment_metrics = any(m in selected_metrics for m in SEGMENT_METRIC_KEYS)
        compute_tortuosity = compute_segment_metrics
        
        for r_idx, seg_list in enumerate(seg_lists):
            seg_metrics = []
            seg_masks = []
            segment_entries = []
            total_geo = 0.0
            sum_curv = 0.0
            sum_curv2 = 0.0
            root_info = root_infos[r_idx] if r_idx < len(root_infos) else _build_root_info(ROOT_CRITERIA[r_idx], None, node_radius_map)

            for seg_idx, seg in enumerate(seg_list, start=1):
                pts = np.array([Gc.nodes[n]['pos'] for n in seg])
                sm = {}
                L_geo = sum(np.linalg.norm(pts[i] - pts[i+1]) for i in range(len(pts)-1))
                sm['geodesic_length'] = L_geo
                total_geo += L_geo
                sm['avg_diameter'] = np.mean([2 * node_radius_map.get(n, 0.5) for n in seg])

                if compute_tortuosity:
                    counts_arr = np.array([seg_counts[r_idx].get(n, 1) for n in seg])
                    tort = compute_tortuosity_metrics(pts, smoothing=0, n_samples=500, counts=counts_arr)
                    sm.update(tort)
                    if not np.isnan(tort['spline_mean_curvature']):
                        sum_curv += tort['spline_mean_curvature'] * L_geo
                    if not np.isnan(tort['spline_mean_square_curvature']):
                        sum_curv2 += tort['spline_mean_square_curvature'] * L_geo
                else:
                    for key in SEGMENT_METRIC_KEYS:
                        if key not in sm:
                            sm[key] = np.nan

                seg_metrics.append(sm)
                segment_entries.append({
                    'segment_index': seg_idx,
                    'root_node': seg[0] if seg else None,
                    'endpoint_node': seg[-1] if seg else None,
                    'node_path': list(seg),
                    'point_path': pts.tolist(),
                })

            if total_geo > 0 and compute_tortuosity:
                agg_curv.append(sum_curv/total_geo)
                agg_curv2.append(sum_curv2/total_geo)
            else:
                agg_curv.append(np.nan)
                agg_curv2.append(np.nan)

            segments_info.append({
                **root_info,
                'segment_metrics': seg_metrics,
                'segment_masks': seg_masks,
                'segments': segment_entries,
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
        
