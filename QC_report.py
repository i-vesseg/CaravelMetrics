# -*- coding: utf-8 -*-
"""
QC_report.py

Generates a self-contained, interactive HTML quality-control report for each
vessel graph produced by Graph_extractor.py / Metric_extractor.py.

Each report contains:
  * an interactive 3D view of the skeleton graph, the segmentation surface
    (translucent, if the binary mask is supplied) and the extracted segments,
    with click-to-select on the 3D view itself or via the segment table;
  * global / per-component statistics, computed with the *same* functions the
    metric pipeline uses, so the numbers shown here are the numbers in the CSVs;
  * per-segment statistics in a sortable, filterable table.

The report is a single HTML file per subject and needs no server. plotly.js is
either inlined, shared as a sibling file, or pulled from a CDN (--plotlyjs).

Usage
-----
Single subject:
    python QC_report.py --graph <subject>_vessel_data.pkl --out QC/ \
        [--mask mask.nii.gz] [--atlas atlas.nii.gz] [--label_map labels.txt]

Whole results tree (also writes an index.html cohort overview):
    python QC_report.py --results_dir /path/to/Results --out QC/ \
        [--mask_dir /path/to/masks] [--atlas_dir /path/to/parcellations]
"""

import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path

import networkx as nx
import nibabel as nib
import numpy as np

from Metric_extractor import (
    compute_metrics_for_mask,
    compute_tortuosity_metrics,
    extract_branch_segments,
    load_label_map,
    robust_graph_loader,
)

# The new definition has no roots: segments are unique branches between
# consecutive junctions/endpoints, so there is a single bucket.
ROOT_NAMES = ['Branch segments (unique)']


# =========================================================================
#                              SMALL HELPERS
# =========================================================================

def _f(x, nd=4):
    """Round to a plain python float, mapping non-finite values to None-safe NaN."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float('nan')
    if not np.isfinite(v):
        return v  # NaN / +-Inf are valid *JavaScript* literals (see _js_literal)
    return round(v, nd)


def _fl(arr, nd=3):
    """Round a numeric array and return it as a plain python list."""
    return [_f(v, nd) for v in np.asarray(arr).ravel()]


def _js_literal(obj):
    """
    Serialise the payload as a JavaScript object literal.

    Deliberately NOT strict JSON: curvature metrics are legitimately NaN for
    short segments and the tortuosity index can be +Inf, both of which json.dumps
    emits as bare NaN/Infinity tokens that JSON.parse() rejects but a JS
    literal accepts natively. Embedding as a literal keeps those values
    meaningful in the browser instead of forcing them to null.
    """
    return json.dumps(obj, allow_nan=True, separators=(',', ':'))


# =========================================================================
#                        MASK SURFACE (context rendering)
# =========================================================================

def build_mask_mesh(mask_path, target_faces=45000):
    """
    Marching-cubes surface of the binary mask, decimated for browser rendering.

    Returned in the same world (mm) frame as the graph node positions, so the
    two overlay correctly: Graph_extractor stores node 'pos' in world
    coordinates via the (possibly resampled) affine, and the affine's origin
    and orientation are preserved by resample_to_isotropic.
    """
    import vedo

    nii = nib.load(mask_path)
    data = np.asarray(nii.dataobj)
    binary = (data > 0).astype(np.uint8)
    del data

    vol = vedo.Volume(binary)
    mesh = vol.isosurface(value=0.5).apply_transform(nii.affine.tolist())
    del vol, binary

    n_faces = mesh.ncells
    if n_faces > target_faces:
        # decimate() takes the FRACTION of faces to keep.
        mesh = mesh.decimate(fraction=float(target_faces) / n_faces)

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.cells, dtype=np.int64)
    if faces.size == 0 or verts.size == 0:
        return None
    # isosurface() emits triangles; guard anyway so a stray quad can't corrupt i/j/k.
    faces = np.array([f for f in faces if len(f) == 3], dtype=np.int64)

    return {
        'x': _fl(verts[:, 0], 2), 'y': _fl(verts[:, 1], 2), 'z': _fl(verts[:, 2], 2),
        'i': faces[:, 0].tolist(), 'j': faces[:, 1].tolist(), 'k': faces[:, 2].tolist(),
        'n_faces_full': int(n_faces), 'n_faces_shown': int(len(faces)),
    }


# =========================================================================
#                          ATLAS REGION ASSIGNMENT
# =========================================================================

def assign_regions(G, node_list, atlas_path, label_map_path):
    """
    Map each graph node to an atlas region id, as Metric_extractor does:
    nibabel/RAS lookup of the node's voxel, any positive label is a region.

    Returns (region_per_node, region_names, n_unassigned).
    """
    img = nib.load(atlas_path)
    atlas = np.round(img.get_fdata(dtype=np.float32)).astype(np.int16)
    inv_affine = np.linalg.inv(img.affine)
    shape = atlas.shape

    labels = load_label_map(label_map_path)

    pos = np.array([G.nodes[n]['pos'] for n in node_list], dtype=np.float64)
    idx = np.round(nib.affines.apply_affine(inv_affine, pos)).astype(int)

    inside = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
    region = np.zeros(len(node_list), dtype=int)
    if np.any(inside):
        ii = idx[inside]
        # Same rule as the metric pipeline: any positive label is a region.
        region[inside] = np.maximum(atlas[ii[:, 0], ii[:, 1], ii[:, 2]], 0)

    names = {int(r): labels.get(int(r), f"region_{int(r)}")
             for r in np.unique(region) if r > 0}
    return region.tolist(), names, int(np.sum(region == 0))


# =========================================================================
#                              PAYLOAD BUILDER
# =========================================================================

def subject_from_pkl(graph_pkl_path):
    """'<subject>_vessel_data.pkl' -> '<subject>'."""
    subject = os.path.basename(graph_pkl_path)
    for suffix in ('_vessel_data.pkl', '.pkl'):
        if subject.endswith(suffix):
            return subject[: -len(suffix)]
    return subject


def build_payload(graph_pkl_path, mask_path=None, atlas_path=None,
                  label_map_path=None, mesh_faces=45000, with_mesh=True,
                  max_segments_per_root=None, line_width=3.0):
    """Collect everything the HTML report needs into one JSON-able dict."""
    subject = subject_from_pkl(graph_pkl_path)

    print(f"  [QC] {subject}: loading graph ...")
    G, node_radius_map = robust_graph_loader(graph_pkl_path)

    node_list = list(G.nodes())
    if not node_list:
        # An empty graph means extraction produced nothing; fail loudly rather
        # than emitting a report full of NaNs that looks like a real result.
        raise ValueError(
            f"{subject}: the graph in {graph_pkl_path} has no nodes — "
            "graph extraction produced an empty result, there is nothing to QC."
        )

    nid2idx = {n: i for i, n in enumerate(node_list)}
    pos = np.array([G.nodes[n]['pos'] for n in node_list], dtype=np.float64)
    radius = np.array([float(node_radius_map.get(n, np.nan)) for n in node_list])
    degree = np.array([G.degree(n) for n in node_list], dtype=int)

    # --- component ids ------------------------------------------------------
    # nx.connected_components() is deterministic for an unmodified graph, so the
    # cid indices produced here match those compute_metrics_for_mask() assigns.
    comp_sets = list(nx.connected_components(G))
    comp_of_node = np.full(len(node_list), -1, dtype=int)
    for cid, cnodes in enumerate(comp_sets):
        for n in cnodes:
            comp_of_node[nid2idx[n]] = cid

    edges = list(G.edges())
    eu = np.array([nid2idx[u] for u, v in edges], dtype=int)
    ev = np.array([nid2idx[v] for u, v in edges], dtype=int)
    elen = np.array([float(G[u][v].get('weight', np.nan)) for u, v in edges])

    # --- regions ------------------------------------------------------------
    region_per_node, region_names, n_unassigned = None, {}, 0
    if atlas_path and os.path.exists(atlas_path):
        print(f"  [QC] {subject}: assigning atlas regions ...")
        region_per_node, region_names, n_unassigned = assign_regions(
            G, node_list, atlas_path, label_map_path
        )

    # --- component-level metrics (identical code path to the CSV pipeline) ---
    print(f"  [QC] {subject}: computing component metrics ...")
    comp_metrics = compute_metrics_for_mask(G, node_radius_map)

    # --- segments -----------------------------------------------------------
    print(f"  [QC] {subject}: extracting segments ...")
    segments = []
    components = []
    for cid, cnodes in enumerate(comp_sets):
        Gc = G.subgraph(cnodes)
        cm = comp_metrics.get(cid, {})

        # Unique branch decomposition: each anatomical branch appears once, so
        # there are no roots and no trunk overlap. Everything is placed in a
        # single bucket (index 0) so the existing report template renders it.
        seg_list = extract_branch_segments(Gc)

        endpoints = [n for n in cnodes if G.degree(n) == 1]
        bifs = [n for n in cnodes if G.degree(n) >= 3]

        roots = []  # no roots with the unique branch decomposition

        if max_segments_per_root is not None and len(seg_list) > max_segments_per_root:
            seg_list = sorted(
                seg_list,
                key=lambda s: sum(
                    np.linalg.norm(G.nodes[s[i]]['pos'] - G.nodes[s[i + 1]]['pos'])
                    for i in range(len(s) - 1)
                ),
                reverse=True,
            )[:max_segments_per_root]

        seg_ids_by_root = [[]]
        for seg in seg_list:
            pts = np.array([G.nodes[n]['pos'] for n in seg], dtype=np.float64)
            l_geo = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0
            chord = float(np.linalg.norm(pts[-1] - pts[0])) if len(pts) > 1 else 0.0
            tort = compute_tortuosity_metrics(pts)
            # Length-weighted mean diameter along the branch.
            diam = np.array([2 * node_radius_map.get(n, 0.5) for n in seg], dtype=np.float64)
            if len(pts) > 1:
                seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
                w = np.zeros(len(seg)); w[:-1] += seg_lens / 2.0; w[1:] += seg_lens / 2.0
                avg_diam = float(np.average(diam, weights=w)) if w.sum() > 0 else float(diam.mean())
            else:
                avg_diam = float(diam.mean())

            sid = len(segments)
            seg_ids_by_root[0].append(sid)
            segments.append({
                'id': sid,
                'comp': cid,
                'root': 0,
                'nodes': [nid2idx[n] for n in seg],
                'n_nodes': len(seg),
                'geodesic_length': _f(l_geo),
                'avg_diameter': _f(avg_diam),
                # No overlap by construction -- every branch edge belongs to
                # exactly one segment.
                'mean_shared_by': 1.0,
                'arc': _f(l_geo),
                'chord': _f(chord),
                'aoc': _f((l_geo / chord) if chord > 0 else float('inf')),
                'kappa': _f(tort['spline_mean_curvature'], 5),
                'kappa2': _f(tort['spline_mean_square_curvature'], 6),
                'rms_kappa': _f(tort['spline_rms_curvature'], 5),
            })

        agg = cm.get('aggregated_tortuosity', {})
        components.append({
            'cid': cid,
            'n_nodes': len(cnodes),
            'n_endpoints': len(endpoints),
            'n_bifurcations': len(bifs),
            'total_length': _f(cm.get('total_length', np.nan), 2),
            'num_bifurcations': int(cm.get('num_bifurcations', 0) or 0),
            'bifurcation_density': _f(cm.get('bifurcation_density', np.nan), 5),
            'volume': _f(cm.get('volume', np.nan), 2),
            'fractal_dimension': _f(cm.get('fractal_dimension', np.nan)),
            'lacunarity': _f(cm.get('lacunarity', np.nan)),
            'num_loops': int(cm.get('num_loops', 0) or 0),
            'num_abnormal_degree_nodes': int(cm.get('num_abnormal_degree_nodes', 0) or 0),
            'roots': [nid2idx[r] for r in roots] if roots else [],
            'root_collision': False,
            'agg_curv': [_f(agg.get('mean_curvature', np.nan), 5)],
            'agg_curv2': [_f(agg.get('mean_square_curvature', np.nan), 6)],
            'main_curv': _f(np.nan, 5),
            'main_curv2': _f(np.nan, 6),
            'n_segments': len(seg_ids_by_root[0]),
            'segs_by_root': seg_ids_by_root,
        })

    # --- how many segments cross each edge (overlap visualisation) -----------
    edge_key = {}
    for e_i in range(len(edges)):
        a, b = int(eu[e_i]), int(ev[e_i])
        edge_key[(min(a, b), max(a, b))] = e_i
    # Single bucket now (root index 0). With unique branches every edge sits on
    # exactly one segment, so this multiplicity map is 1 everywhere a branch
    # runs -- the visible opposite of the old root-path overlap.
    seg_mult = np.zeros((len(edges), 1), dtype=int)
    for s in segments:
        nds = s['nodes']
        for a, b in zip(nds[:-1], nds[1:]):
            e_i = edge_key.get((min(a, b), max(a, b)))
            if e_i is not None:
                seg_mult[e_i, 0] += 1

    # --- global aggregation (matches global_summary.csv arithmetic) ---------
    total_length = float(np.nansum([c['total_length'] for c in components]))
    glob = {
        'n_nodes': len(node_list),
        'n_edges': len(edges),
        'n_components': len(comp_sets),
        'total_length': _f(total_length, 2),
        'num_bifurcations': int(sum(c['num_bifurcations'] for c in components)),
        'num_endpoints': int(np.sum(degree == 1)),
        'volume_frustum': _f(np.nansum([c['volume'] for c in components]), 2),
        'num_loops': int(sum(c['num_loops'] for c in components)),
        'num_abnormal_degree_nodes': int(sum(c['num_abnormal_degree_nodes'] for c in components)),
        'n_segments': len(segments),
    }
    glob['bifurcation_density'] = _f(glob['num_bifurcations'] / total_length, 5) if total_length > 0 else float('nan')
    if components:
        biggest = max(components, key=lambda c: (c['total_length'] if np.isfinite(c['total_length']) else 0))
        glob['largest_comp_length_frac'] = _f(
            (biggest['total_length'] / total_length) if total_length > 0 else np.nan, 4
        )
        glob['fractal_dimension_largest'] = biggest['fractal_dimension']
        glob['lacunarity_largest'] = biggest['lacunarity']

    # --- true mask volume ----------------------------------------------------
    mask_info = None
    if mask_path and os.path.exists(mask_path):
        m_img = nib.load(mask_path)
        m_data = np.asarray(m_img.dataobj) > 0
        vox_vol = float(np.prod(m_img.header.get_zooms()[:3]))
        mask_info = {
            'path': mask_path,
            'shape': [int(s) for s in m_img.shape[:3]],
            'zooms': [_f(z, 4) for z in m_img.header.get_zooms()[:3]],
            'voxel_volume_mm3': _f(vox_vol, 5),
            'n_voxels': int(np.sum(m_data)),
            'true_volume': _f(float(np.sum(m_data)) * vox_vol, 2),
        }
        glob['volume_true'] = mask_info['true_volume']
        del m_data

    # --- mesh ----------------------------------------------------------------
    mesh = None
    if with_mesh and mask_path and os.path.exists(mask_path):
        print(f"  [QC] {subject}: building mask surface ...")
        try:
            mesh = build_mask_mesh(mask_path, target_faces=mesh_faces)
        except Exception as exc:  # rendering context is optional, never fatal
            print(f"  [QC] ! mask surface failed ({exc}); continuing without it.")

    payload = {
        'subject': subject,
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'sources': {
            'graph': os.path.abspath(graph_pkl_path),
            'mask': os.path.abspath(mask_path) if mask_path and os.path.exists(mask_path) else None,
            'atlas': os.path.abspath(atlas_path) if atlas_path and os.path.exists(atlas_path) else None,
        },
        'nodes': {
            'x': _fl(pos[:, 0]), 'y': _fl(pos[:, 1]), 'z': _fl(pos[:, 2]),
            'r': _fl(radius), 'deg': degree.tolist(), 'comp': comp_of_node.tolist(),
            'region': region_per_node,
        },
        'edges': {
            'u': eu.tolist(), 'v': ev.tolist(), 'len': _fl(elen),
            'mult': seg_mult.T.tolist(),  # [root][edge]
        },
        'components': components,
        'segments': segments,
        'global': glob,
        'regions': {str(k): v for k, v in region_names.items()},
        'n_nodes_unassigned_region': n_unassigned,
        'mask': mask_info,
        'mesh': mesh,
        'root_names': ROOT_NAMES,
        'default_line_width': float(line_width),
    }
    return payload


# =========================================================================
#                              HTML RENDERING
# =========================================================================

def _plotly_script_tag(mode, out_dir):
    """Return the <script> tag for plotly.js, writing a sibling bundle if needed."""
    if mode == 'cdn':
        return '<script src="https://cdn.plot.ly/plotly-3.0.1.min.js" charset="utf-8"></script>'
    if mode == 'inline':
        from plotly.offline import get_plotlyjs
        return f'<script type="text/javascript">{get_plotlyjs()}</script>'
    # 'directory': one shared bundle next to the reports (default -- keeps a
    # cohort of reports small without needing network access).
    bundle = os.path.join(out_dir, 'plotly.min.js')
    if not os.path.exists(bundle):
        from plotly.offline import get_plotlyjs
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        # Write-then-rename: concurrent subjects (e.g. a ProcessPoolExecutor
        # pool) can race on this same path; os.replace is atomic on POSIX,
        # so readers never see a half-written bundle.
        tmp = f"{bundle}.tmp-{os.getpid()}"
        with open(tmp, 'w', encoding='utf-8') as fh:
            fh.write(get_plotlyjs())
        os.replace(tmp, bundle)
    return '<script src="plotly.min.js" charset="utf-8"></script>'


def render_html(payload, plotly_tag):
    return (
        HTML_TEMPLATE
        .replace('__PLOTLY_JS__', plotly_tag)
        .replace('__SUBJECT__', payload['subject'])
        .replace('__DATA__', _js_literal(payload))
    )


def generate_qc_report(graph_pkl_path, output_dir, mask_path=None, atlas_path=None,
                       label_map_path=None, plotlyjs='directory', mesh_faces=45000,
                       with_mesh=True, max_segments_per_root=None, overwrite=True,
                       line_width=3.0):
    """
    Build one interactive QC HTML for one graph pickle.

    Returns (html_path, payload); payload is None when an existing report was
    kept because overwrite=False. The existence check runs before build_payload
    so skipping actually avoids the work rather than only the final write.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_html = os.path.join(output_dir, f"{subject_from_pkl(graph_pkl_path)}_QC.html")
    if os.path.exists(out_html) and not overwrite:
        print(f"  [QC] exists, skipping: {out_html}")
        return out_html, None

    payload = build_payload(
        graph_pkl_path, mask_path=mask_path, atlas_path=atlas_path,
        label_map_path=label_map_path, mesh_faces=mesh_faces, with_mesh=with_mesh,
        max_segments_per_root=max_segments_per_root, line_width=line_width,
    )
    tag = _plotly_script_tag(plotlyjs, output_dir)
    with open(out_html, 'w', encoding='utf-8') as fh:
        fh.write(render_html(payload, tag))
    size_mb = os.path.getsize(out_html) / 1e6
    print(f"  [QC] -> {out_html}  ({size_mb:.1f} MB)")
    return out_html, payload


# =========================================================================
#                          BATCH / COHORT INDEX
# =========================================================================

def _find_sibling(directory, subject_id, patterns):
    """First existing file matching any of `patterns` formatted with the subject id."""
    if not directory:
        return None
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(directory, pat.format(s=subject_id))))
        if hits:
            return hits[0]
    return None


def resolve_mask(graph_pkl_path, subject, mask_dir):
    """Explicit --mask_dir wins; otherwise fall back to the path stored in the pickle."""
    subject_id = subject[:-4] if subject.endswith('-MRA') else subject
    hit = _find_sibling(mask_dir, subject_id, ['{s}.nii.gz', '{s}-MRA.nii.gz', '{s}*.nii.gz'])
    if hit:
        return hit
    try:
        import pickle
        with open(graph_pkl_path, 'rb') as fh:
            stored = pickle.load(fh).get('input_file')
        if stored and os.path.exists(stored):
            return stored
    except Exception:
        pass
    return None


def write_index(rows, output_dir):
    """Cohort overview linking every report, with simple outlier shading."""
    index_path = os.path.join(output_dir, 'index.html')
    html = (
        INDEX_TEMPLATE
        .replace('__DATA__', _js_literal(rows))
        .replace('__GENERATED__', datetime.now().strftime('%Y-%m-%d %H:%M'))
    )
    with open(index_path, 'w', encoding='utf-8') as fh:
        fh.write(html)
    print(f"\n  [QC] cohort index -> {index_path}")
    return index_path


def run_batch(results_dir, output_dir, mask_dir=None, atlas_dir=None, label_map_path=None,
              plotlyjs='directory', mesh_faces=45000, with_mesh=True,
              max_segments_per_root=None, overwrite=True, line_width=3.0):
    pkls = sorted(glob.glob(os.path.join(results_dir, '**', '*_vessel_data.pkl'), recursive=True))
    print(f"Found {len(pkls)} graph pickle(s) under {results_dir}")

    rows = []
    for pkl in pkls:
        subject = os.path.basename(pkl).replace('_vessel_data.pkl', '')
        subject_id = subject[:-4] if subject.endswith('-MRA') else subject
        mask = resolve_mask(pkl, subject, mask_dir)
        atlas = _find_sibling(atlas_dir, subject_id,
                              ['{s}_ArterialAtlas_in_MRA_dilated.nii.gz',
                               '{s}_ArterialAtlas_in_MRA.nii.gz',
                               '{s}*parcellation*dilated.nii.gz',
                               '{s}*parcellation*.nii.gz'])
        if atlas is None:
            # Reports written next to the graph keep the registered atlas alongside it.
            atlas = _find_sibling(os.path.dirname(pkl), subject,
                                  ['{s}_registered_parcellation_dilated.nii.gz',
                                   '{s}_registered_parcellation.nii.gz'])
        try:
            _, payload = generate_qc_report(
                pkl, output_dir, mask_path=mask, atlas_path=atlas,
                label_map_path=label_map_path, plotlyjs=plotlyjs, mesh_faces=mesh_faces,
                with_mesh=with_mesh, max_segments_per_root=max_segments_per_root,
                overwrite=overwrite, line_width=line_width,
            )
        except Exception as exc:
            print(f"  [QC] ! FAILED {subject}: {exc}")
            rows.append({'subject': subject, 'error': str(exc)})
            continue

        if payload is None:
            # Kept an existing report; its metrics were never recomputed, so the
            # index can link it but cannot fill in the cohort columns.
            rows.append({'subject': subject, 'href': f'{subject}_QC.html'})
            continue

        g = payload['global']
        rows.append({
            'subject': payload['subject'],
            'href': f"{payload['subject']}_QC.html",
            'total_length': g['total_length'],
            'num_bifurcations': g['num_bifurcations'],
            'bifurcation_density': g['bifurcation_density'],
            'volume': g.get('volume_true', g['volume_frustum']),
            'n_components': g['n_components'],
            'largest_frac': g.get('largest_comp_length_frac', float('nan')),
            'num_loops': g['num_loops'],
            'n_abnormal': g['num_abnormal_degree_nodes'],
            'fractal_dimension': g.get('fractal_dimension_largest', float('nan')),
            'n_segments': g['n_segments'],
        })

    if rows:
        write_index(rows, output_dir)
    return rows


# =========================================================================
#                               TEMPLATES
# =========================================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QC (branch segments) — __SUBJECT__</title>
__PLOTLY_JS__
<style>
  :root {
    --bg:#ffffff; --panel:#ffffff; --panel2:#f4f5f6; --line:#d7dbe0;
    --fg:#1a1d21; --muted:#6b7280; --accent:#26415c; --warn:#9a5b12;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }
  h1,h2,h3 { margin:0 0 .4em; font-weight:600; }
  h1 { font-size:18px; letter-spacing:.01em; }
  h2 { font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.09em;
       padding-bottom:6px; border-bottom:1px solid var(--line); margin-bottom:10px; }
  a { color:var(--accent); }
  header { padding:14px 20px; background:var(--panel); border-bottom:1px solid var(--line);
           display:flex; flex-wrap:wrap; align-items:baseline; gap:16px; }
  header .meta { color:var(--muted); font-size:11.5px; }
  .wrap { padding:16px 20px 40px; max-width:1900px; margin:0 auto; }
  .cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:10px; margin-bottom:22px; }
  .card { background:var(--panel); border:1px solid var(--line); padding:10px 12px; }
  .card .lab { color:var(--muted); font-size:10.5px; text-transform:uppercase; letter-spacing:.05em; }
  .card .val { font-size:18px; font-weight:600; margin-top:2px; }
  .card .sub { color:var(--muted); font-size:11px; }
  .main { display:grid; grid-template-columns:minmax(0,1fr) 300px; gap:14px; align-items:start; }
  @media (max-width:1150px){ .main{grid-template-columns:1fr} }
  .box { background:var(--panel); border:1px solid var(--line); padding:12px; }
  #view { height:74vh; min-height:520px; }
  .ctl { display:flex; flex-direction:column; gap:12px; max-height:74vh; overflow:auto; }
  .grp { border-bottom:1px solid var(--line); padding-bottom:10px; }
  .grp:last-child{border-bottom:none;padding-bottom:0}
  label { display:block; color:var(--muted); font-size:10.5px; text-transform:uppercase;
          letter-spacing:.05em; margin-bottom:4px; }
  select, input[type=text] { width:100%; background:var(--panel); color:var(--fg);
          border:1px solid var(--line); border-radius:2px; padding:5px 7px; font-size:12.5px; }
  input[type=range]{width:100%}
  .chk { display:flex; align-items:center; gap:7px; color:var(--fg); font-size:12.5px;
         text-transform:none; letter-spacing:0; margin:3px 0; cursor:pointer; }
  .chk input{margin:0; cursor:pointer}
  .btnrow { display:flex; gap:0; flex-wrap:wrap; }
  button { background:var(--panel); color:var(--fg); border:1px solid var(--line);
           padding:5px 10px; font-size:12px; cursor:pointer; margin-right:-1px; }
  button:hover{background:var(--panel2)}
  button.on{background:var(--fg);color:#fff;border-color:var(--fg);position:relative;z-index:1}
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th,td { padding:5px 7px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }
  th:first-child,td:first-child{text-align:left}
  th { position:sticky; top:0; background:var(--panel2); cursor:pointer; user-select:none;
       color:var(--muted); font-weight:600; font-size:10.5px; text-transform:uppercase; }
  th:hover{color:var(--fg)}
  tbody tr{cursor:pointer} tbody tr:hover{background:var(--panel2)}
  tbody tr.sel{background:#eef2f6; box-shadow:inset 2px 0 0 var(--accent)}
  td.flagged{color:var(--warn)} td.nan{color:var(--muted)}
  .scroll { max-height:430px; overflow:auto; }
  .note { color:var(--muted); font-size:12px; margin:6px 0 0; }
  .legend { display:flex; flex-wrap:wrap; gap:5px; margin-top:6px; max-height:150px; overflow:auto; }
  .lg { display:flex; align-items:center; gap:5px; font-size:11px; color:var(--muted); }
  .sw { width:9px; height:9px; }
  code { background:var(--panel2); padding:1px 5px; font-size:11.5px; }
</style>
</head>
<body>
<header>
  <h1>Vessel graph QC — <span style="color:var(--accent)">__SUBJECT__</span></h1>
  <span style="background:#0f766e;color:#fff;font-size:11px;font-weight:600;padding:3px 9px;border-radius:10px;letter-spacing:.03em">NEW · branch decomposition</span>
  <span class="meta" id="meta"></span>
</header>

<div class="wrap">
  <h2>Global statistics</h2>
  <div class="cards" id="cards"></div>

  <h2>3D inspection</h2>
  <div class="main">
    <div class="box"><div id="view"></div></div>

    <div class="box ctl">
      <div class="grp">
        <label>Colour the skeleton by</label>
        <select id="colorby">
          <option value="uniform">Uniform</option>
          <option value="radius">Radius (mm)</option>
          <option value="degree">Node degree</option>
          <option value="component">Connected component</option>
          <option value="mult">Segment multiplicity (overlap)</option>
          <option value="region">Atlas region</option>
        </select>
        <div class="legend" id="legend"></div>
      </div>

      <div class="grp">
        <label>Segments</label>
        <select id="root"></select>
        <select id="comp" style="margin-top:6px"></select>
        <div class="btnrow" style="margin-top:8px">
          <button id="mAll" class="on">All coloured</button>
          <button id="mSel">Selected only</button>
          <button id="mNone">Hide</button>
        </div>
        <p class="note">Click any segment in the 3D view to select it.</p>
      </div>

      <div class="grp">
        <label>Line width</label>
        <input type="range" id="lineWidth" min="1" max="7" step="0.5" value="3">
      </div>

      <div class="grp">
        <label>Overlays</label>
        <div class="chk"><input type="checkbox" id="cMesh" checked><span>Mask surface</span></div>
        <input type="range" id="opac" min="0" max="60" value="24">
        <div class="chk"><input type="checkbox" id="cEnd"><span>Endpoints (degree 1)</span></div>
        <div class="chk"><input type="checkbox" id="cBif"><span>Bifurcations (degree ≥3)</span></div>
        <div class="chk"><input type="checkbox" id="cAbn"><span>Abnormal degree (&gt;3)</span></div>
        <div class="chk"><input type="checkbox" id="cZero"><span>Zero-radius nodes</span></div>
        <div class="chk"><input type="checkbox" id="cRoot"><span>Segment roots</span></div>
      </div>

      <div class="grp">
        <label>Selected segment</label>
        <div id="seginfo" class="note">Click a row in the table below.</div>
      </div>
    </div>
  </div>

  <h2 style="margin-top:20px">Segments <span id="segcount" style="color:var(--muted);text-transform:none;letter-spacing:0"></span></h2>
  <div class="box">
    <div class="btnrow" style="margin-bottom:8px">
      <input type="text" id="filter" placeholder="filter…" style="flex:1;min-width:160px">
      <button id="onlyFlag">Flagged only</button>
    </div>
    <div class="scroll"><table id="segtable"><thead></thead><tbody></tbody></table></div>
    <p class="note">Segments are <b>unique branches</b> — the maximal centerline chain between two
    consecutive junctions/endpoints (the VMTK / literature definition). They do <b>not</b> overlap:
    every edge belongs to exactly one segment, so the branch lengths sum to the component's
    total length and <code>shared</code> is 1 by construction. <code>a/c</code> = geodesic
    path length / chord (tortuosity index), <code>κ</code> = spline mean curvature (1/mm),
    <code>κ²</code> = mean-square curvature.</p>
  </div>

  <h2 style="margin-top:20px">Components</h2>
  <div class="box"><div class="scroll"><table id="comptable"><thead></thead><tbody></tbody></table></div></div>
</div>

<script>
const D = __DATA__;

/* ---------------------------------------------------------------- helpers */
const N = D.nodes, E = D.edges;
const fmt = (v, n=2) => (v===null||v===undefined||Number.isNaN(v)) ? '—'
      : (!Number.isFinite(v) ? '∞' : (Math.abs(v)>=1000 ? v.toFixed(0) : v.toFixed(n)));
const VIRIDIS = ['#440154','#482878','#3e4989','#31688e','#26828e','#1f9e89','#35b779','#6ece58','#b5de2b','#fde725'];
// Matplotlib "tab10"-style qualitative palette: desaturated enough to read
// cleanly on a white background, unlike the neon set this replaced.
const CAT20 = ['#4c72b0','#dd8452','#55a868','#c44e52','#8172b1','#937860','#da8bc3','#8c8c8c','#ccb974','#64b5cd',
               '#3b6299','#c76a37','#3d7a51','#a83a3e','#6a5a94','#7a6350','#b56ea3','#707070','#a89a55','#4d94ab'];
const lerpColor = t => VIRIDIS[Math.max(0, Math.min(VIRIDIS.length-1, Math.round(t*(VIRIDIS.length-1))))];

/* Node-value arrays used for the colour modes. */
function nodeValues(mode) {
  if (mode === 'radius')    return N.r;
  if (mode === 'degree')    return N.deg;
  if (mode === 'component') return N.comp;
  if (mode === 'region')    return N.region;
  return null;
}

/* --------------------------------------------------------------- state */
const S = {
  colorby: 'uniform',
  root: 0,
  comp: 'all',              // 'all' pools every component's segments for the current root
  segMode: 'all',           // 'sel' | 'all' | 'none' -- 'all' by default so
                             // there is something on screen to click straight away
  selected: null,
  mesh: true, opacity: 0.24,
  lineWidth: D.default_line_width || 3,
  endpoints: false, bifs: false, abnormal: false, zero: false, roots: false,
  onlyFlagged: false, filter: '',
  sortKey: 'geodesic_length', sortDir: -1,
  // Start from a fixed oblique view so every report in a cohort opens at the
  // same orientation and subjects can be compared without re-aiming the camera.
  camera: {eye:{x:1.4,y:-1.5,z:1.1}, up:{x:0,y:0,z:1}},
};

// Dark slate centerline: reads as a technical line drawing on white, unlike
// the neon accent this replaced.
const SKEL_COLOR = '#37424d';

const segFlagged = s => !Number.isFinite(s.kappa) || (Number.isFinite(s.aoc) && s.aoc > 3)
                        || s.n_nodes < 4;

/* --------------------------------------------------- 3D trace construction */
function edgeLineTrace(idxs, color, width, name, hover) {
  const x=[], y=[], z=[];
  for (const e of idxs) {
    const u=E.u[e], v=E.v[e];
    x.push(N.x[u], N.x[v], null); y.push(N.y[u], N.y[v], null); z.push(N.z[u], N.z[v], null);
  }
  return { type:'scatter3d', mode:'lines', x, y, z, name,
           line:{color, width}, hoverinfo: hover?'text':'skip',
           showlegend:false, ...(hover?{text:hover}:{}) };
}

function pathTrace(nodes, color, width, name, segId) {
  return { type:'scatter3d', mode:'lines', name, showlegend:false, hoverinfo:'skip',
           x:nodes.map(i=>N.x[i]), y:nodes.map(i=>N.y[i]), z:nodes.map(i=>N.z[i]),
           line:{color, width},
           ...(segId!==undefined ? {customdata:nodes.map(()=>segId)} : {}) };
}

function markerTrace(idxs, color, size, name, symbol='circle') {
  return { type:'scatter3d', mode:'markers', name, showlegend:false,
           x:idxs.map(i=>N.x[i]), y:idxs.map(i=>N.y[i]), z:idxs.map(i=>N.z[i]),
           marker:{color, size, symbol, line:{width:0}},
           hoverinfo:'text',
           text: idxs.map(i=>`node ${i}<br>deg ${N.deg[i]}<br>r ${fmt(N.r[i],2)} mm<br>comp ${N.comp[i]}`) };
}

/* Skeleton, split into colour bins computed in the browser so the coordinate
   arrays are shipped once rather than once per colour mode. */
function skeletonTraces() {
  const nE = E.u.length;
  const all = [...Array(nE).keys()];
  const lw = S.lineWidth;
  if (S.colorby === 'uniform') {
    return [edgeLineTrace(all, SKEL_COLOR, lw, 'skeleton')];
  }
  if (S.colorby === 'mult') {
    // Multiplicity is extremely skewed -- peripheral twigs sit on 1 segment while
    // the shared proximal trunk can sit on every segment of the root (100+). Bin
    // on log scale so the low end stays readable instead of collapsing into one
    // colour at the bottom of a linear ramp.
    const m = E.mult[S.root];
    let mx = 0; for (let e=0;e<nE;e++) if (m[e]>mx) mx = m[e];
    const NB = 9, lmax = Math.log(Math.max(2, mx));
    const bins = Array.from({length:NB+1},()=>[]);   // slot 0 = not on any segment
    for (let e=0;e<nE;e++) {
      if (m[e] <= 0) { bins[0].push(e); continue; }
      const b = Math.min(NB-1, Math.floor(Math.log(m[e])/lmax*(NB-1e-9)));
      bins[b+1].push(e);
    }
    const edgeOf = b => Math.round(Math.exp(b/NB*lmax));
    LEGEND = [];
    const traces = [];
    bins.forEach((idxs,b)=>{
      if (!idxs.length) return;
      const col = b===0 ? '#d7dbe0' : lerpColor((b-1)/(NB-1));
      LEGEND.push({c:col, l: b===0 ? 'not on any segment'
                    : `${edgeOf(b-1)}–${edgeOf(b)}×`});
      traces.push(edgeLineTrace(idxs, col, b===0?Math.max(1,lw-1):lw+0.5, `mult ${b}`));
    });
    return traces;
  }
  const vals = nodeValues(S.colorby);
  if (!vals) return [edgeLineTrace(all, SKEL_COLOR, lw, 'skeleton')];

  if (S.colorby === 'component' || S.colorby === 'region') {
    const bins = new Map();
    for (let e=0;e<nE;e++){ const k=vals[E.u[e]]; if(!bins.has(k)) bins.set(k,[]); bins.get(k).push(e); }
    const keys=[...bins.keys()].sort((a,b)=>a-b);
    LEGEND = keys.map((k,i)=>({
      c: (S.colorby==='region' && k===0) ? '#d7dbe0' : CAT20[i%CAT20.length],
      l: S.colorby==='region' ? (k===0?'unassigned':(D.regions[k]||('region '+k))) : ('comp '+k)
    }));
    return keys.map((k,i)=>edgeLineTrace(bins.get(k),
        (S.colorby==='region' && k===0) ? '#d7dbe0' : CAT20[i%CAT20.length], lw, String(k)));
  }
  /* continuous: radius / degree */
  let lo=Infinity, hi=-Infinity;
  for (let e=0;e<nE;e++){ const t=0.5*(vals[E.u[e]]+vals[E.v[e]]); if(t<lo)lo=t; if(t>hi)hi=t; }
  const NB=10, bins=Array.from({length:NB},()=>[]);
  for (let e=0;e<nE;e++){
    const t=0.5*(vals[E.u[e]]+vals[E.v[e]]);
    let b = hi>lo ? Math.floor((t-lo)/(hi-lo)*(NB-1e-9)) : 0;
    bins[Math.max(0,Math.min(NB-1,b))].push(e);
  }
  LEGEND = bins.map((_,b)=>({c:lerpColor(b/(NB-1)), l:fmt(lo+(hi-lo)*b/NB, 2)}));
  return bins.map((idxs,b)=>edgeLineTrace(idxs, lerpColor(b/(NB-1)), lw, 'bin'+b));
}

let LEGEND = [];

// S.comp is either a specific component's cid or the sentinel 'all', meaning
// every component's segments for the current root are pooled together.
function currentSegIds() {
  if (S.comp === 'all') {
    let ids = [];
    for (const c of D.components) ids = ids.concat(c.segs_by_root[S.root] || []);
    return ids;
  }
  const comp = D.components.find(c=>c.cid===S.comp);
  return comp ? (comp.segs_by_root[S.root] || []) : [];
}

function segmentTraces() {
  const out = [];
  if (S.segMode === 'none') return out;
  const ids = currentSegIds();

  if (S.segMode === 'all') {
    // Group by palette slot instead of one trace per segment: a component can
    // have 175+ segments, and 175 separate scatter3d traces stalls the renderer.
    // customdata carries the segment id per point (null at the null-separated
    // breaks) so a click on any point can be traced back to its segment.
    const groups = Array.from({length:CAT20.length},()=>({x:[],y:[],z:[],cd:[]}));
    ids.forEach((sid,i)=>{
      const g = groups[i%CAT20.length], s = D.segments[sid];
      for (const n of s.nodes){ g.x.push(N.x[n]); g.y.push(N.y[n]); g.z.push(N.z[n]); g.cd.push(sid); }
      g.x.push(null); g.y.push(null); g.z.push(null); g.cd.push(null);
    });
    groups.forEach((g,i)=>{ if(g.x.length) out.push({
      type:'scatter3d', mode:'lines', x:g.x, y:g.y, z:g.z, customdata:g.cd, showlegend:false,
      hoverinfo:'skip', line:{color:CAT20[i], width:3}, name:'seg'+i }); });
  }
  if (S.selected !== null) {
    const s = D.segments[S.selected];
    if (s && (S.comp === 'all' || s.comp === S.comp) && s.root === S.root) {
      out.push(pathTrace(s.nodes, '#c0392b', 7, 'selected', s.id));
      out.push(markerTrace([s.nodes[0]], '#1a1d21', 6, 'root'));
      out.push(markerTrace([s.nodes[s.nodes.length-1]], '#c0392b', 6, 'tip'));
    }
  }
  return out;
}

function overlayTraces() {
  const out = [];
  const nN = N.x.length;
  if (S.endpoints) {
    const idx=[]; for(let i=0;i<nN;i++) if(N.deg[i]===1) idx.push(i);
    out.push(markerTrace(idx, '#4c72b0', 2.5, 'endpoints'));
  }
  if (S.bifs) {
    const idx=[]; for(let i=0;i<nN;i++) if(N.deg[i]>=3) idx.push(i);
    out.push(markerTrace(idx, '#55a868', 3, 'bifurcations'));
  }
  if (S.abnormal) {
    const idx=[]; for(let i=0;i<nN;i++) if(N.deg[i]>3) idx.push(i);
    if (idx.length) out.push(markerTrace(idx, '#dd8452', 4, 'degree>3', 'diamond'));
  }
  if (S.zero) {
    const idx=[]; for(let i=0;i<nN;i++) if(!(N.r[i]>0)) idx.push(i);
    if (idx.length) out.push(markerTrace(idx, '#c44e52', 4, 'zero radius', 'x'));
  }
  if (S.roots) {
    const comps = S.comp === 'all' ? D.components : D.components.filter(c=>c.cid===S.comp);
    const idx = comps.filter(c=>c.roots.length).map(c=>c.roots[S.root]);
    if (idx.length) out.push(markerTrace(idx, '#1a1d21', 8, 'root', 'diamond'));
  }
  return out;
}

function meshTrace() {
  if (!D.mesh || !S.mesh) return [];
  return [{ type:'mesh3d', x:D.mesh.x, y:D.mesh.y, z:D.mesh.z,
            i:D.mesh.i, j:D.mesh.j, k:D.mesh.k,
            color:'#a9b4bf', opacity:S.opacity, flatshading:false,
            hoverinfo:'skip', showscale:false, name:'mask' }];
}

const gd = document.getElementById('view');

// Fixed scene bounds computed once from the full dataset (skeleton + mesh, if
// present), not from whichever traces happen to be toggled on. Plotly
// auto-ranges each scene axis to fit only the currently visible traces, so
// without this, toggling an overlay checkbox changes the axis range and the
// view appears to "jump" even though the camera itself never moved -- a
// smaller trace set (e.g. mesh off) auto-zooms in, a larger one zooms back out.
function computeBounds() {
  let xlo=Infinity,xhi=-Infinity, ylo=Infinity,yhi=-Infinity, zlo=Infinity,zhi=-Infinity;
  const upd = (x,y,z) => {
    if (Number.isFinite(x)) { if (x<xlo) xlo=x; if (x>xhi) xhi=x; }
    if (Number.isFinite(y)) { if (y<ylo) ylo=y; if (y>yhi) yhi=y; }
    if (Number.isFinite(z)) { if (z<zlo) zlo=z; if (z>zhi) zhi=z; }
  };
  for (let i=0;i<N.x.length;i++) upd(N.x[i], N.y[i], N.z[i]);
  if (D.mesh) for (let i=0;i<D.mesh.x.length;i++) upd(D.mesh.x[i], D.mesh.y[i], D.mesh.z[i]);
  const pad = 0.04;
  const span = (lo,hi) => { const d=(hi-lo)*pad || 1; return [lo-d, hi+d]; };
  return { x: span(xlo,xhi), y: span(ylo,yhi), z: span(zlo,zhi) };
}
const BOUNDS = computeBounds();

function baseLayout() {
  // visible:false drops grid, ticks, axis lines and background planes entirely
  // -- just the graph floating on a plain canvas, no scientific-looking chart junk.
  // range/autorange:false pins the coordinate frame to BOUNDS regardless of
  // which traces are currently drawn (see computeBounds above).
  const ax = rng => ({ visible:false, showspikes:false, range:rng, autorange:false });
  return { paper_bgcolor:'#ffffff', plot_bgcolor:'#ffffff',
           margin:{l:0,r:0,t:0,b:0}, showlegend:false,
           // uirevision makes Plotly.react preserve camera/zoom across redraws
           // on its own; combined with the fixed range above this is belt-and-braces.
           uirevision:'keep',
           scene:{ xaxis:ax(BOUNDS.x), yaxis:ax(BOUNDS.y), zaxis:ax(BOUNDS.z), aspectmode:'data',
                   uirevision:'keep', ...(S.camera ? {camera:S.camera} : {}) } };
}

function draw() {
  const traces = [...meshTrace(), ...skeletonTraces(), ...segmentTraces(), ...overlayTraces()];
  Plotly.react(gd, traces, baseLayout(), {responsive:true, displaylogo:false});
  renderLegend();
}

// Registering these listeners is deferred to the boot section at the bottom
// of this script, right after the first draw() call. Plotly only attaches its
// event emitter (the .on method) to `gd` during its first Plotly.react/newPlot
// call -- attaching listeners here, before that first call has happened,
// silently no-ops (the `gd.on && ...` guard just skips it) and the camera
// then appears to "snap back" on every redraw because S.camera never gets
// updated from the user's actual mouse-drag orbit interactions.
function attachGraphListeners() {
  gd.on('plotly_relayout', ev => { if (ev['scene.camera']) S.camera = ev['scene.camera']; });

  // Click-to-select: any traced segment point carries its segment id as
  // customdata (see segmentTraces), so a click anywhere on a visible segment
  // selects it without needing the table.
  gd.on('plotly_click', ev => {
    const pt = ev.points && ev.points[0];
    if (!pt || typeof pt.customdata !== 'number') return;
    S.selected = pt.customdata;
    if (S.segMode === 'none') setSegMode('sel');
    renderSegTable(); renderSegInfo(); draw();
  });
}

function renderLegend() {
  const el = document.getElementById('legend');
  if (!LEGEND.length || S.colorby==='uniform') { el.innerHTML=''; return; }
  el.innerHTML = LEGEND.map(g=>`<span class="lg"><span class="sw" style="background:${g.c}"></span>${g.l}</span>`).join('');
}

/* -------------------------------------------------------------- header UI */
document.getElementById('meta').textContent =
  `generated ${D.generated} · graph ${D.sources.graph}` +
  (D.sources.mask ? ` · mask ${D.sources.mask.split('/').pop()}` : ' · no mask supplied') +
  (D.sources.atlas ? ` · atlas ${D.sources.atlas.split('/').pop()}` : '');

const G = D.global;
const cards = [
  ['Total length', fmt(G.total_length,1), 'mm'],
  ['Vessel volume', fmt(G.volume_true ?? G.volume_frustum,1),
     (G.volume_true!==undefined ? 'mm³ (voxel-counted)' : 'mm³ (frustum model)')],
  ['Bifurcations', G.num_bifurcations, `density ${fmt(G.bifurcation_density,4)} /mm`],
  ['Endpoints', G.num_endpoints, 'degree-1 nodes'],
  ['Components', G.n_components, `largest = ${fmt((G.largest_comp_length_frac??NaN)*100,1)}% of length`],
  ['Nodes / edges', `${G.n_nodes} / ${G.n_edges}`, ''],
  ['Loops', G.num_loops, 'independent cycles'],
  ['Degree &gt;3 nodes', G.num_abnormal_degree_nodes, 'skeleton crossings'],
  ['Fractal dim.', fmt(G.fractal_dimension_largest,3), 'largest component'],
  ['Lacunarity', fmt(G.lacunarity_largest,3), 'largest component'],
  ['Segments', G.n_segments, 'unique branches'],
];
document.getElementById('cards').innerHTML = cards.map(
  c=>`<div class="card"><div class="lab">${c[0]}</div><div class="val">${c[1]}</div><div class="sub">${c[2]}</div></div>`
).join('');

/* ------------------------------------------------------------- selectors */
const rootSel = document.getElementById('root');
rootSel.innerHTML = D.root_names.map((n,i)=>`<option value="${i}">${n}</option>`).join('');

const compSel = document.getElementById('comp');
const compsSorted = [...D.components].sort((a,b)=>(b.total_length||0)-(a.total_length||0));
compSel.innerHTML = `<option value="all">All components (${compsSorted.length})</option>` + compsSorted.map(c=>
  `<option value="${c.cid}">Component ${c.cid} — ${fmt(c.total_length,1)} mm, ${c.n_nodes} nodes, ${(c.segs_by_root[0]||[]).length} segs</option>`
).join('');
compSel.value = 'all';   // S.comp already defaults to 'all'

rootSel.onchange = e => { S.root=+e.target.value; S.selected=null; draw(); renderSegTable(); };
compSel.onchange = e => { S.comp = e.target.value==='all' ? 'all' : +e.target.value; S.selected=null; draw(); renderSegTable(); };
document.getElementById('colorby').onchange = e => {
  S.colorby = e.target.value;
  // With every component's segments shown at once, the "All coloured" overlay
  // covers nearly the whole tree and completely hides any skeleton colouring
  // drawn underneath it -- so a colour choice here would look like it did
  // nothing. Hide that overlay automatically so the chosen colouring is
  // actually visible; segments can be brought back with the buttons below.
  if (S.colorby !== 'uniform' && S.segMode === 'all') setSegMode('none');
  draw();
};

const SEG_MODE_BTN = {sel:'mSel', all:'mAll', none:'mNone'};
function setSegMode(mode) {
  S.segMode = mode;
  Object.entries(SEG_MODE_BTN).forEach(([m,id])=>document.getElementById(id).classList.toggle('on', m===mode));
}
for (const [id,mode] of [['mSel','sel'],['mAll','all'],['mNone','none']]) {
  document.getElementById(id).onclick = () => { setSegMode(mode); draw(); };
}
const bind = (id,key) => { const el=document.getElementById(id);
  el.onchange = () => { S[key]=el.checked; draw(); }; };
bind('cMesh','mesh'); bind('cEnd','endpoints'); bind('cBif','bifs');
bind('cAbn','abnormal'); bind('cZero','zero'); bind('cRoot','roots');
document.getElementById('opac').oninput = e => { S.opacity = +e.target.value/100; draw(); };
document.getElementById('lineWidth').oninput = e => { S.lineWidth = +e.target.value; draw(); };
if (!D.mesh) { document.getElementById('cMesh').disabled = true;
               document.getElementById('cMesh').checked = false; S.mesh=false;
               document.getElementById('opac').disabled = true; }
if (!N.region) { [...document.querySelectorAll('#colorby option')]
                  .find(o=>o.value==='region').disabled = true; }

/* ---------------------------------------------------------- segment table */
const SEG_COLS = [
  ['id','#',0], ['comp','comp',0], ['n_nodes','nodes',0], ['geodesic_length','length (mm)',2],
  ['avg_diameter','Ø (mm)',2], ['chord','chord (mm)',2],
  ['aoc','a/c',3], ['kappa','κ (1/mm)',4], ['kappa2','κ²',5],
  ['rms_kappa','rms κ',4], ['mean_shared_by','shared',1],
];

function currentSegments() {
  let rows = currentSegIds().map(i=>D.segments[i]);
  if (S.onlyFlagged) rows = rows.filter(segFlagged);
  if (S.filter) { const f=S.filter.toLowerCase();
    rows = rows.filter(s=>SEG_COLS.some(c=>String(s[c[0]]).toLowerCase().includes(f))); }
  const k=S.sortKey, d=S.sortDir;
  return rows.sort((a,b)=>{
    const av=a[k], bv=b[k];
    const an=!Number.isFinite(av), bn=!Number.isFinite(bv);
    if (an&&bn) return 0; if (an) return 1; if (bn) return -1;   // NaN always last
    return (av-bv)*d;
  });
}

function renderSegTable() {
  const rows = currentSegments();
  const t = document.getElementById('segtable');
  t.querySelector('thead').innerHTML = '<tr>' + SEG_COLS.map(
    c=>`<th data-k="${c[0]}">${c[1]}${S.sortKey===c[0]?(S.sortDir>0?' ▲':' ▼'):''}</th>`).join('') + '</tr>';
  t.querySelector('tbody').innerHTML = rows.map(s=>{
    const fl = segFlagged(s);
    return `<tr data-id="${s.id}" class="${S.selected===s.id?'sel':''}">` + SEG_COLS.map(c=>{
      const v = s[c[0]];
      const cls = !Number.isFinite(v) ? 'nan' : (fl && (c[0]==='aoc'||c[0]==='kappa') ? 'flagged' : '');
      return `<td class="${cls}">${c[2]===0?(v??'—'):fmt(v,c[2])}</td>`;
    }).join('') + '</tr>';
  }).join('');

  t.querySelectorAll('th').forEach(th=>th.onclick=()=>{
    const k=th.dataset.k;
    if (S.sortKey===k) S.sortDir*=-1; else { S.sortKey=k; S.sortDir=-1; }
    renderSegTable();
  });
  t.querySelectorAll('tbody tr').forEach(tr=>tr.onclick=()=>{
    S.selected = +tr.dataset.id;
    if (S.segMode==='none') setSegMode('sel');
    renderSegTable(); renderSegInfo(); draw();
  });

  const total = currentSegIds().length;
  const where = S.comp === 'all' ? `all ${D.components.length} components` : `component ${S.comp}`;
  document.getElementById('segcount').textContent =
    `— showing ${rows.length} of ${total} branches for ${where}`;
}

function renderSegInfo() {
  const el = document.getElementById('seginfo');
  if (S.selected===null) { el.textContent='Click a row in the table below.'; return; }
  const s = D.segments[S.selected];
  el.innerHTML =
    `<b style="color:var(--accent)">Branch ${s.id}</b> · component ${s.comp}<br>` +
    `${s.n_nodes} nodes · ${fmt(s.geodesic_length,2)} mm · Ø ${fmt(s.avg_diameter,2)} mm<br>` +
    `path ${fmt(s.arc,2)} / chord ${fmt(s.chord,2)} = <b>${fmt(s.aoc,3)}</b><br>` +
    `κ ${fmt(s.kappa,4)} 1/mm · κ² ${fmt(s.kappa2,5)} · rms κ ${fmt(s.rms_kappa,4)}<br>` +
    `<span style="color:var(--muted)">unique branch — no shared trunk</span>` +
    (segFlagged(s) ? '<br><span style="color:var(--warn)">⚠ flagged: short segment or extreme path/chord</span>' : '');
}

document.getElementById('filter').oninput = e => { S.filter=e.target.value; renderSegTable(); };
document.getElementById('onlyFlag').onclick = function(){
  S.onlyFlagged=!S.onlyFlagged; this.classList.toggle('on',S.onlyFlagged); renderSegTable(); };

/* -------------------------------------------------------- component table */
const COMP_COLS = [
  ['cid','component',0],['n_nodes','nodes',0],['total_length','length (mm)',1],
  ['num_bifurcations','bifurc.',0],['n_endpoints','endpoints',0],
  ['bifurcation_density','bif/mm',4],['volume','volume (mm³)',1],
  ['fractal_dimension','fractal',3],['lacunarity','lacunarity',3],
  ['num_loops','loops',0],['num_abnormal_degree_nodes','deg>3',0],
];
(function renderCompTable(){
  const t = document.getElementById('comptable');
  t.querySelector('thead').innerHTML='<tr>'+COMP_COLS.map(c=>`<th>${c[1]}</th>`).join('')+'</tr>';
  t.querySelector('tbody').innerHTML = compsSorted.map(c=>
    `<tr data-cid="${c.cid}">`+COMP_COLS.map(col=>{
      const v=c[col[0]];
      return `<td class="${Number.isFinite(v)||col[2]===0?'':'nan'}">${col[2]===0?v:fmt(v,col[2])}</td>`;
    }).join('')+'</tr>').join('');
  t.querySelectorAll('tbody tr').forEach(tr=>tr.onclick=()=>{
    S.comp=+tr.dataset.cid; S.selected=null; compSel.value=S.comp;
    draw(); renderSegTable(); renderSegInfo();
    document.getElementById('view').scrollIntoView({behavior:'smooth',block:'center'});
  });
})();

/* -------------------------------------------------------------------- boot */
document.getElementById('lineWidth').value = S.lineWidth;
draw();
attachGraphListeners();
renderSegTable();
renderSegInfo();
</script>
</body>
</html>
"""


INDEX_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Caravel QC — cohort overview</title>
<style>
  :root{--bg:#ffffff;--panel:#ffffff;--panel2:#f4f5f6;--line:#d7dbe0;--fg:#1a1d21;
        --muted:#6b7280;--accent:#26415c;--warn:#9a5b12;}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
    font:13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
  header{padding:14px 20px;background:var(--panel);border-bottom:1px solid var(--line)}
  h1{margin:0;font-size:17px} .meta{color:var(--muted);font-size:11.5px}
  .wrap{padding:16px 20px 60px;max-width:1700px;margin:0 auto}
  .box{background:var(--panel);border:1px solid var(--line);padding:12px}
  table{width:100%;border-collapse:collapse;font-size:12.5px}
  th,td{padding:6px 8px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
  th:first-child,td:first-child{text-align:left}
  th{position:sticky;top:0;background:var(--panel2);cursor:pointer;color:var(--muted);
     font-size:10.5px;text-transform:uppercase;letter-spacing:.05em}
  th:hover{color:var(--fg)} tbody tr:hover{background:var(--panel2)}
  a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
  td.out{color:var(--warn);font-weight:600}
  .note{color:var(--muted);font-size:12px;margin-top:10px}
  input{background:var(--panel);color:var(--fg);border:1px solid var(--line);
        border-radius:2px;padding:6px 9px;font-size:13px;width:260px;margin-bottom:10px}
</style>
</head>
<body>
<header>
  <h1>Caravel QC — cohort overview</h1>
  <div class="meta">generated __GENERATED__ · click a subject to open its report ·
       values more than 2 SD from the cohort mean are highlighted</div>
</header>
<div class="wrap">
  <input id="q" placeholder="filter subjects…">
  <div class="box"><table id="t"><thead></thead><tbody></tbody></table></div>
</div>
<script>
const ROWS = __DATA__;
const COLS = [
  ['subject','subject',null],['total_length','total length (mm)',1],
  ['volume','volume (mm³)',1],['num_bifurcations','bifurcations',0],
  ['bifurcation_density','bif/mm',4],['n_components','components',0],
  ['largest_frac','largest frac',3],['num_loops','loops',0],['n_abnormal','deg>3',0],
  ['fractal_dimension','fractal',3],['n_segments','segments',0],
];
const fmt=(v,n)=>(v===null||v===undefined||Number.isNaN(v))?'—':(n===0?v:Number(v).toFixed(n));

// Flag cohort outliers so a bad subject stands out without opening every report.
const stats={};
for (const [k,,d] of COLS) { if(d===null) continue;
  const v=ROWS.map(r=>r[k]).filter(Number.isFinite);
  if(!v.length) continue;
  const m=v.reduce((a,b)=>a+b,0)/v.length;
  stats[k]={m, s:Math.sqrt(v.reduce((a,b)=>a+(b-m)**2,0)/v.length)};
}
let sortKey='subject', sortDir=1, filter='';
function render(){
  const t=document.getElementById('t');
  t.querySelector('thead').innerHTML='<tr>'+COLS.map(c=>
    `<th data-k="${c[0]}">${c[1]}${sortKey===c[0]?(sortDir>0?' ▲':' ▼'):''}</th>`).join('')+'</tr>';
  let rows=ROWS.filter(r=>r.subject.toLowerCase().includes(filter));
  rows.sort((a,b)=>{const x=a[sortKey],y=b[sortKey];
    if(typeof x==='string') return x.localeCompare(y)*sortDir;
    const xn=!Number.isFinite(x), yn=!Number.isFinite(y);
    if(xn&&yn)return 0; if(xn)return 1; if(yn)return -1; return (x-y)*sortDir;});
  t.querySelector('tbody').innerHTML=rows.map(r=>{
    if(r.error) return `<tr><td>${r.subject}</td><td colspan="${COLS.length-1}" style="color:var(--warn);text-align:left">${r.error}</td></tr>`;
    return '<tr>'+COLS.map(c=>{
      if(c[0]==='subject') return `<td><a href="${r.href}">${r.subject}</a></td>`;
      const v=r[c[0]], st=stats[c[0]];
      const out = st && st.s>0 && Number.isFinite(v) && Math.abs(v-st.m)>2*st.s;
      return `<td class="${out?'out':''}">${fmt(v,c[2])}</td>`;
    }).join('')+'</tr>';
  }).join('');
  t.querySelectorAll('th').forEach(th=>th.onclick=()=>{
    const k=th.dataset.k; if(sortKey===k) sortDir*=-1; else {sortKey=k;sortDir=1;} render();});
}
document.getElementById('q').oninput=e=>{filter=e.target.value.toLowerCase();render();};
render();
</script>
</body>
</html>
"""


# =========================================================================
#                             CLI ENTRY POINT
# =========================================================================

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_LABEL_MAP = os.path.join(SCRIPT_DIR, "Atlas", "ArterialAtlasLables.txt")

    parser = argparse.ArgumentParser(
        description="Generate interactive HTML QC reports for Caravel vessel graphs.")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--graph", help="Path to a single <subject>_vessel_data.pkl.")
    src.add_argument("--results_dir", help="Directory tree to scan for *_vessel_data.pkl.")

    parser.add_argument("--out", required=True, help="Output directory for the HTML reports.")
    parser.add_argument("--mask", default=None,
                        help="Binary vessel mask NIfTI for the translucent surface and the "
                             "voxel-counted volume (single-graph mode).")
    parser.add_argument("--atlas", default=None,
                        help="Registered arterial atlas NIfTI for region colouring (single-graph mode).")
    parser.add_argument("--mask_dir", default=None,
                        help="Directory of masks, matched to subjects by name (batch mode).")
    parser.add_argument("--atlas_dir", default=None,
                        help="Directory of registered atlases, matched by name (batch mode).")
    parser.add_argument("--label_map", default=DEFAULT_LABEL_MAP,
                        help="ArterialAtlasLables.txt for region names.")

    parser.add_argument("--plotlyjs", choices=['directory', 'inline', 'cdn'], default='directory',
                        help="How to ship plotly.js: 'directory' (default) writes one shared file "
                             "next to the reports; 'inline' embeds it in each file (~4.9 MB); "
                             "'cdn' needs internet.")
    parser.add_argument("--mesh_faces", type=int, default=45000,
                        help="Target triangle count for the decimated mask surface (default: 45000).")
    parser.add_argument("--no_mesh", action="store_true",
                        help="Skip the mask surface even when a mask is available.")
    parser.add_argument("--max_segments_per_root", type=int, default=None,
                        help="Cap segments stored per root, keeping the longest (default: all).")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Do not regenerate reports that already exist.")
    parser.add_argument("--line_width", type=float, default=3.0,
                        help="Initial skeleton line width shown in the report; adjustable "
                             "afterwards with the in-page slider (default: 3.0).")

    args = parser.parse_args()

    if args.results_dir:
        run_batch(args.results_dir, args.out, mask_dir=args.mask_dir, atlas_dir=args.atlas_dir,
                  label_map_path=args.label_map, plotlyjs=args.plotlyjs,
                  mesh_faces=args.mesh_faces, with_mesh=not args.no_mesh,
                  max_segments_per_root=args.max_segments_per_root,
                  overwrite=not args.skip_existing, line_width=args.line_width)
    else:
        subject = os.path.basename(args.graph).replace('_vessel_data.pkl', '').replace('.pkl', '')
        mask = args.mask or resolve_mask(args.graph, subject, None)
        generate_qc_report(args.graph, args.out, mask_path=mask, atlas_path=args.atlas,
                           label_map_path=args.label_map, plotlyjs=args.plotlyjs,
                           mesh_faces=args.mesh_faces, with_mesh=not args.no_mesh,
                           max_segments_per_root=args.max_segments_per_root,
                           overwrite=not args.skip_existing, line_width=args.line_width)
