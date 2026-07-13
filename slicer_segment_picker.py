# -*- coding: utf-8 -*-
"""
slicer_segment_picker.py

Standalone script meant to be run inside 3D Slicer's Python console (not part
of the CaravelMetrics pipeline, no dependency on it). Enables click-to-inspect
on a segment model exported by Segment_visualizer.export_segments_vtp():
left-clicking the model in the 3D view prints the clicked segment's id,
endpoints, node count and tortuosity metrics to the Python console.

Usage (in Slicer):
  1. Load the .vtp produced by Segment_visualizer.export_segments_vtp() as a
     Model (File > Add Data).
  2. Optionally color it: select the model in the Models module, Display >
     Scalars > 'SegmentID', pick a discrete/categorical colormap.
  3. Open the Python console (View > Python console) and run:
         exec(open(r"<path to this file>").read())
         enable_segment_picking()
  4. Left-click anywhere on the model in the 3D view.

Notes:
  - Picking reads cell-data arrays straight off whatever polydata was
    clicked (vtkCellPicker.GetDataSet()), so no lookup against a specific
    MRML model node is needed. If several models are visible, this reports
    whichever one is under the cursor; clicks that miss every model (empty
    space, or a model without a 'SegmentID' array) are silently ignored.
  - Left-click still rotates/pans the camera as usual (this observer only
    reads the pick result, it doesn't consume the event).
"""
import vtk
import slicer

_METRIC_LABELS = [
    ('ArcLength', 'spline arc length'),
    ('ChordLength', 'spline chord length'),
    ('MeanCurvature', 'mean curvature'),
    ('MeanSquareCurvature', 'mean square curvature'),
    ('RmsCurvature', 'rms curvature'),
    ('ArcOverChord', 'arc / chord'),
    ('FitRmse', 'spline fit RMSE'),
]


def _format_segment_info(cell_data, cell_id):
    def get(name):
        arr = cell_data.GetArray(name)
        return arr.GetValue(cell_id) if arr else None

    seg_id = get('SegmentID')
    lines = [
        f"Segment #{int(seg_id) if seg_id is not None else '?'}",
        f"  endpoints        : {get('EndpointA')} <-> {get('EndpointB')}",
        f"  node count       : {get('NodeCount')}",
        f"  degenerate       : {bool(get('Degenerate'))}",
    ]
    for array_name, label in _METRIC_LABELS:
        value = get(array_name)
        if value is None:
            continue
        value_str = "NaN" if value != value else f"{value:.4f}"  # NaN != NaN
        lines.append(f"  {label:<22}: {value_str}")
    return "\n".join(lines)


def _on_left_click(interactor, event):
    x, y = interactor.GetEventPosition()
    view = slicer.app.layoutManager().threeDWidget(0).threeDView()
    renderer = view.renderWindow().GetRenderers().GetFirstRenderer()

    picker = vtk.vtkCellPicker()
    picker.SetTolerance(0.005)
    picker.Pick(x, y, 0, renderer)

    cell_id = picker.GetCellId()
    if cell_id < 0:
        return

    poly_data = picker.GetDataSet()
    if poly_data is None or not poly_data.GetCellData().GetArray('SegmentID'):
        return  # clicked something that isn't a segment-colored model

    print("\n" + "=" * 50)
    print(_format_segment_info(poly_data.GetCellData(), cell_id))
    print("=" * 50)


def enable_segment_picking():
    """Register the left-click observer on the 3D view. Call once per session."""
    view = slicer.app.layoutManager().threeDWidget(0).threeDView()
    interactor = view.interactorStyle().GetInteractor()
    interactor.AddObserver("LeftButtonPressEvent", _on_left_click, 1.0)
    print("Segment picking enabled: left-click on the segment model in the 3D view.")


if __name__ == "__main__":
    enable_segment_picking()
