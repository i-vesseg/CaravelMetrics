# -*- coding: utf-8 -*-
"""
Segment_visualizer.py

Visualization helpers for the segments produced by
Segment_extractor.build_segments(). Kept separate from Segment_extractor.py:
that module only extracts segments/metrics, this one only exports them for
viewing (currently: a colored .vtp for 3D Slicer / ParaView).

Reuses the same vtk-based polyline export pattern already used elsewhere in
the codebase (Metric_extractor._write_segment_vtp), applied here to the
new build_segments() output instead of the shortest-path segments.
"""
import os
import sys

# See Segment_extractor.py's module top for why this is needed: this file
# lives in coronary_extension/, one level below the repo root (baseline
# CaravelMetrics modules + sibling coronary_extension/ modules), and neither
# is a pip-installed package -- this puts both on sys.path so the flat
# imports in __main__ below resolve whether this module is imported from a
# root script or run directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Maps the VTP cell-array name (Slicer/ParaView friendly) to the
# corresponding key in compute_tortuosity_metrics' output.
# fit_rmse is intentionally excluded: it's computed by evaluating the s=0
# interpolating spline at the same parameter values used to fit it, so the
# residual is structurally ~0 for every segment (see
# Segment_extractor.segments_to_dataframe, which excludes it for the same
# reason).
_METRIC_VTP_ARRAYS = {
    'ArcLength': 'spline_arc_length',
    'ChordLength': 'spline_chord_length',
    'MeanCurvature': 'spline_mean_curvature',
    'MeanSquareCurvature': 'spline_mean_square_curvature',
    'RmsCurvature': 'spline_rms_curvature',
    'ArcOverChord': 'arc_over_chord',
}


def export_segments_vtp(segments, G, output_path):
    """
    Write `segments` (as returned by build_segments(G)) to a .vtp file, one
    polyline per segment, so it can be loaded as a Model in 3D Slicer /
    ParaView. Every cell (= one segment) carries its id, endpoints, node
    count, degenerate flag and tortuosity metrics as per-cell scalar arrays:
      - Coloring: in Slicer, Display > Scalars > 'SegmentID' with a
        categorical/discrete colormap colors each segment distinctly.
      - Click-to-inspect: every array is attached directly to the polydata's
        CellData, so a vtkCellPicker on this model can read a segment's full
        metrics straight off the picked cell, with no lookup table or
        companion file needed.

    Degenerate segments (no metrics computed) get NaN for every metric
    array; VTK float arrays store NaN natively.
    """
    import vtk

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    int_arrays = {}
    for name in ('SegmentID', 'NodeCount', 'Degenerate', 'EndpointA', 'EndpointB'):
        arr = vtk.vtkIntArray()
        arr.SetName(name)
        int_arrays[name] = arr

    float_arrays = {}
    for name in _METRIC_VTP_ARRAYS:
        arr = vtk.vtkFloatArray()
        arr.SetName(name)
        float_arrays[name] = arr

    for seg in segments:
        nodes = seg['nodes']
        if len(nodes) < 2:
            continue

        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(nodes))
        for i, n in enumerate(nodes):
            pos = G.nodes[n]['pos']
            point_id = points.InsertNextPoint(float(pos[0]), float(pos[1]), float(pos[2]))
            polyline.GetPointIds().SetId(i, point_id)
        lines.InsertNextCell(polyline)

        int_arrays['SegmentID'].InsertNextValue(int(seg['id']))
        int_arrays['NodeCount'].InsertNextValue(len(nodes))
        int_arrays['Degenerate'].InsertNextValue(int(seg['degenerate']))
        int_arrays['EndpointA'].InsertNextValue(int(seg['endpoints'][0]))
        int_arrays['EndpointB'].InsertNextValue(int(seg['endpoints'][1]))

        metrics = seg.get('metrics') or {}
        for vtp_name, metric_key in _METRIC_VTP_ARRAYS.items():
            value = metrics.get(metric_key)
            float_arrays[vtp_name].InsertNextValue(float(value) if value is not None else float('nan'))

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetLines(lines)
    for arr in list(int_arrays.values()) + list(float_arrays.values()):
        poly_data.GetCellData().AddArray(arr)
    poly_data.GetCellData().SetActiveScalars('SegmentID')

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(poly_data)
    writer.Write()
    print(f"Saved colored segment VTP: {output_path}")


if __name__ == "__main__":
    # --- Demo: build segments for one Eva_Coronaries subject and export the
    # colored VTP. Reuses Segment_extractor.build_segments() and the same
    # extract_graph()-based loading as its own sanity-check block.
    import os
    import pickle
    import tempfile

    from Graph_extractor import extract_graph
    from Segment_extractor import build_segments

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

    vtp_path = os.path.join(scratch_dir, "vessel_segments_colored.vtp")
    export_segments_vtp(segments, G, vtp_path)
    print(f"\nLoad this file in 3D Slicer (File > Add Data) as a Model, "
          f"then set Display > Scalars to 'SegmentID' to color each segment.")
