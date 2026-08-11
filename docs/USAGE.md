## 💻 Usage

**Contents:** [About](#about) · [Install](#install) · [Running the full pipeline](#running-the-full-pipeline) ·
[Output](#output) · [Running individual stages](#running-individual-stages) · [Batch processing](#batch-processing) ·
[QC reports](#qc-reports) · [Testing](#running-the-test-suite)

---

## About

CaravelMetrics was built to analyze the **brain** cerebrovascular network: starting from a TOF-MRA
angiography image and its vessel segmentation mask, it registers an arterial atlas (30 territories,
Liu et al. 2023) to get territory-level metrics on top of whole-graph ones. That's the pipeline
described in [Running the full pipeline](#running-the-full-pipeline) below, and the default atlas
bundled in `Atlas/` is brain-specific.

Underneath, though, graph extraction and metric computation don't know anything about brains --
they operate on any binary vessel mask, or any vessel graph, regardless of anatomy. So the same
code has also been used on other vascular structures (e.g. coronary arteries) by skipping
registration entirely -- either by passing `--no_atlas` to `CaravelMetrics.py`, or by starting
directly from a mask or a pre-built graph, see [Running individual stages](#running-individual-stages).

---

## Install

```bash
git clone https://github.com/i-vesseg/CaravelMetrics.git
cd CaravelMetrics
pip install -r requirements.txt
```

FSL (`flirt`, `bet`, `robustfov`, `fslreorient2std`, `convert_xfm`, `fslmaths`) must be installed
and on `PATH` separately -- see [INSTALLATION.md](INSTALLATION.md) -- but only if you register to
an atlas. Graph extraction and whole-graph metrics don't need it.

---

## Running the full pipeline

`CaravelMetrics.py` is the single-command entry point: given a subject's MRA + vessel mask (+
optional T1), it runs registration, graph extraction, and metric computation in one call --
or skips registration with `--no_atlas`, see [below](#without-atlas-whole-graph-only).

### With atlas (regional analysis)

```bash
python CaravelMetrics.py \
  --image  subject001_MRA.nii.gz \
  --mask   subject001_vessel_mask.nii.gz \
  --t1     subject001_T1.nii.gz \
  --mra_stripped --t1_stripped \
  --output_dir results/ \
  --qc
```

If your T1/MRA aren't already skull-stripped, drop `--mra_stripped`/`--t1_stripped` (BET +
`robustfov` run automatically). If you have no T1 at all, drop `--t1`: registration falls back to
warping the MNI template directly onto the MRA. Preprocessing flags (`--denoise`, `--debias`,
`--t1_debias`) and `--registration_qc` (a PNG snapshot of the registered labels, separate from
`--qc`'s HTML graph report) are optional -- run `python CaravelMetrics.py --help` for the full list.

### Without atlas (whole-graph only)

Pass `--no_atlas` to skip registration -- and the FSL dependency it requires -- while still
running graph extraction and metric computation in one call:

```bash
python CaravelMetrics.py \
  --image  subject001_MRA.nii.gz \
  --mask   subject001_vessel_mask.nii.gz \
  --output_dir results/ \
  --no_atlas
```

`--image` is still required (it's used to name the subject's output folder), but no MRA/atlas/T1
arguments are read. Metrics land in a sibling `_GLOBAL` folder instead of the atlas layout -- see
[Output](#output) below.

You can also chain graph extraction and metric computation yourself, which is exactly what
`--no_atlas` does internally, minus the registration step in front. This is the way to go if you
don't have an MRA/reference image at all, or you're working from a mask produced elsewhere:

```bash
python Graph_extractor.py \
  --mask subject001_vessel_mask.nii.gz \
  --output_dir results/

python Metric_extractor.py \
  --graph results/subject001_vessel_mask_vessel_data.pkl \
  --out_dir results/ \
  --mask subject001_vessel_mask.nii.gz
```

---

## Output

**With atlas**, `results/subject001_MRA/` contains the registered/dilated atlas, the vessel graph
(`.pkl` + `.vtp`), `subject001_MRA_global_summary.csv` (whole-graph metrics),
`subject001_MRA_region_summary.csv` + `subject001_MRA_all_components_by_region.csv`
(per-territory), plus (with `--qc`) an interactive QC report -- every file is prefixed with the
subject name, so nothing collides if you ever gather several subjects' folders into one place:

```
results/
└── subject001_MRA/
    ├── subject001_MRA_vessel_data.pkl                        # graph: networkx + per-node radius
    ├── subject001_MRA_vessel_graph.vtp                       # ParaView / ITK-SNAP
    ├── subject001_MRA_vessel_graph_slicer.vtp                # 3D Slicer (LPS/RAS flip applied)
    ├── subject001_MRA_registered_parcellation_dilated.nii.gz
    ├── subject001_MRA_global_summary.csv                     # whole-graph metrics
    ├── subject001_MRA_region_summary.csv                     # per-territory aggregated metrics
    ├── subject001_MRA_all_components_by_region.csv           # per-connected-component metrics
    ├── subject001_MRA_QC.html                                # QC report (with --qc)
    ├── regional_vtps/                                        # one VTP per atlas territory
    ├── regional_results/                                     # per-region, per-component detail + segment CSVs
    └── processing.log
```

The QC report defaults into the subject's own folder alongside everything else; pass `--qc_dir` to
put it somewhere else instead (e.g. a shared `QC/` folder across subjects for a batch run).

**Without atlas** (`--no_atlas`, or the manual `Graph_extractor.py`/`Metric_extractor.py` chain),
there's no per-territory breakdown and no `global_summary.csv` -- metrics go to a sibling
`<output_folder>_GLOBAL/` directory as `region_summary.csv` (with a single `global` pseudo-region)
+ `all_components_by_region.csv`, both still prefixed with the subject name. This "GLOBAL" split is
`extract_metrics`'s convention whenever `registered_atlas_path` is `None`, so it applies the same
way whether you ran `CaravelMetrics.py --no_atlas` or the individual stages yourself. E.g.
`CaravelMetrics.py --no_atlas --output_dir results/ --image subject001_MRA.nii.gz ...` writes the
vessel graph (and QC report, with `--qc`) to `results/subject001_MRA/` but metrics to
`results_GLOBAL/subject001_MRA/`.

---

## Running individual stages

Each stage -- registration, graph extraction, metric computation -- can be called on its own, both
as a CLI and as a plain Python function. This is how the pipeline extends beyond brain MRA: if you
already have a mask, or already have a graph, you don't need registration or graph extraction at
all, and nothing here is brain-specific.

### Already have a mask

Same as [Without atlas](#without-atlas-whole-graph-only) above: run `Graph_extractor.py` then
`Metric_extractor.py` directly. Works for any binary vessel mask, any anatomy or modality (TOF-MRA,
CTA, ...) -- registration and the arterial atlas are brain/MRA-specific, but graph extraction and
metrics are not.

### Already have a graph

Skip graph extraction too. `Metric_extractor.extract_metrics` doesn't require a mask if you already
have a vessel graph -- `robust_graph_loader` auto-detects Caravel's own pickle format, a raw
NetworkX graph, a UNC Skwerer/Bullitt tree dictionary, or a MetaIO/TubeTK `.tre` scene, so any of
these work as `--graph`/`graph_pkl_path` directly. This is the path for structures from other
vessel-extraction tools (e.g. a coronary artery tree exported as `.tre`):

```bash
python Metric_extractor.py --graph my_vessel_tree.pkl --out_dir results/
python Metric_extractor.py --graph my_vessel_tree.tre --out_dir results/
```

```python
from Metric_extractor import extract_metrics

extract_metrics(
    output_folder="results/",
    graph_pkl_path="my_vessel_tree.tre",
    selected_metrics=["total_length", "num_bifurcations", "spline_mean_curvature", "num_loops"],
)
```

### Registration only

Useful for pre-computing atlases for a cohort ahead of a batch graph/metrics run.

```bash
python Atlas_Registration.py \
  --image subject001_MRA.nii.gz \
  --t1    subject001_T1.nii.gz \
  --mra_stripped --t1_stripped \
  --output_dir results/registration/
```

```python
from Atlas_Registration import process_registration

registered_atlas_path, inv_matrix_path = process_registration(
    image_path="subject001_MRA.nii.gz",
    output_dir="results/registration/",
    image_t1_path="subject001_T1.nii.gz",
    mra_stripped=True,
    t1_stripped=True,
)
```

---

## Batch processing

There's no built-in batch runner (each subject's data layout tends to differ), but the pattern is
short -- parallelize with `ProcessPoolExecutor` over subjects, reusing `extract_graph`'s built-in
skip-if-already-processed check to make reruns cheap:

```python
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from Graph_extractor import extract_graph
from Metric_extractor import extract_metrics

MASK_DIR, OUTPUT_DIR = "data/masks", "results/cohort"
METRICS = ["total_length", "volume", "num_bifurcations", "bifurcation_density", "num_loops"]

def process_subject(mask_path):
    subject_id = Path(mask_path).name.split(".")[0]
    out_dir = Path(OUTPUT_DIR, subject_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_pkl = extract_graph(nifti_path=mask_path, output_folder=str(out_dir))
    extract_metrics(output_folder=str(out_dir), graph_pkl_path=graph_pkl,
                     selected_metrics=METRICS, mask_path=mask_path)

if __name__ == "__main__":
    masks = glob.glob(f"{MASK_DIR}/*.nii.gz")
    with ProcessPoolExecutor(max_workers=4) as ex:
        list(ex.map(process_subject, masks))
```

For atlas-based batch runs, register each subject first (this is the slow, FSL-dependent step) and
reuse the registered atlas path across reruns the same way.

---

## QC reports

Two independent QC outputs, either usable standalone:

- **Registration QC** (`Atlas_Registration.generate_qc_snapshot`, or `--qc`/`--registration_qc` on
  the CLIs above): a static PNG grid of sagittal/coronal/axial slices with the registered atlas
  contoured on top of the MRA.
- **Graph QC** (`QC_report.generate_qc_report`, or `--qc` on `CaravelMetrics.py`): an interactive
  HTML report (3D graph viewer, per-component/per-region metric tables) built directly from a
  `vessel_data.pkl`:

```python
from QC_report import generate_qc_report

generate_qc_report(
    graph_pkl_path="results/subject001_vessel_mask_vessel_data.pkl",
    output_dir="results/QC/",
    mask_path="subject001_vessel_mask.nii.gz",
)
```

---

## Running the test suite

The repo ships a small pytest smoke-test suite with its own bundled sample data (not meant as
usage examples -- see [tests/test_pipeline.py](../tests/test_pipeline.py)) that exercises the full
pipeline end-to-end, including the brain full-pipeline flow, whole-graph-only on non-brain masks,
and metrics-only on external graph formats:

```bash
pip install pytest
pytest tests/ -v
```

Results are written to `tests/output/` (gitignored) so you can inspect them after the run.

---

## Reference: input data requirements

- **Vessel mask** (required): binary NIfTI (1 = vessel, 0 = background), any modality (TOF-MRA,
  CTA, ...), any anatomy.
- **MRA/reference image**: the intensity image the mask was segmented from. Needed for
  registration; `CaravelMetrics.py` still requires `--image` even with `--no_atlas` (it names the
  subject's output folder), but `Graph_extractor.py`/`Metric_extractor.py` run without it.
- **T1 image** (optional, atlas mode): improves registration accuracy via an
  MNI-T1-to-subject-T1-to-subject-MRA chain instead of registering the MNI template directly to
  the MRA. Omit it and registration still works.
- **Arterial atlas + label map** (optional, atlas mode): defaults to the bundled
  `Atlas/ArterialAtlas.nii.gz` (Liu et al. 2023, 30 territories) and
  `Atlas/ArterialAtlasLables.txt`; pass `--parcellation`/`--label_map` (or
  `mni_parcellation`/`label_map_path`) for a different atlas.
