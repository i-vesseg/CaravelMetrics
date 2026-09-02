# 💻 Usage

## Install

```bash
git clone https://github.com/i-vesseg/CaravelMetrics.git
cd CaravelMetrics
pip install -r requirements.txt
```

FSL (`flirt`, `bet`, `robustfov`, `fslreorient2std`, `convert_xfm`, `fslmaths`) must be installed
and on `PATH` separately -- see [INSTALLATION.md](INSTALLATION.md) -- but only if you register to
an atlas. Graph extraction and whole-graph metrics don't need it.



## Pipeline at a glance

The pipeline is four stages. `CaravelMetrics.py` runs them end to end for one subject, or you can
call each stage on its own (see [Running individual stages](#running-individual-stages)).

| # | Stage | Script | Input → Output |
|---|-------|--------|----------------|
| 1 | Atlas registration *(optional)* | `Atlas_Registration.py` | MRA (+T1) → registered atlas in MRA space |
| 2 | Graph extraction | `Graph_extractor.py` | vessel mask → vessel graph (`.pkl` + `.vtp`) |
| 3 | Metric computation | `Metric_extractor.py` | graph (+atlas, +mask) → metric CSVs |
| 4 | QC report *(optional)* | `QC_report.py` | graph (+mask, +atlas) → interactive HTML |

Stages 2–3 are the **core**: they give whole-graph metrics for any binary vessel mask, any anatomy
or modality, with no atlas and no FSL. Stage 1 is an **optional layer** -- it's only needed to add
the per-territory (regional) breakdown, and it's the only part that requires an MRA and FSL.



## Running the full pipeline

`CaravelMetrics.py` runs the core pipeline for one subject in a single call: **mask → graph →
metrics**, producing whole-graph morphometrics. The arterial atlas is an **optional layer on
top** -- when enabled, it additionally reports the same metrics broken down per vascular territory
(regional analysis). The atlas layer is the only part that requires an MRA and FSL.

Note that the atlas layer runs **by default**: it is disabled with `--no_atlas`, rather than
enabled with an opt-in flag. The `--html_qc` report and QC snapshots are optional in both cases.

### Core + atlas layer — regional analysis (default)

```bash
python CaravelMetrics.py \
  --image  subject001_MRA.nii.gz \
  --mask   subject001_vessel_mask.nii.gz \
  --t1     subject001_T1.nii.gz \
  --mra_stripped --t1_stripped \
  --output_dir results/ \
  --html_qc --registration_snapshot
```

If the T1/MRA are not already skull-stripped, omit `--mra_stripped`/`--t1_stripped` (BET +
`robustfov` run automatically). If no T1 is available, omit `--t1`: registration falls back to
warping the MNI template directly onto the MRA. The two QC outputs are independent: `--html_qc`
writes the interactive HTML graph report, `--registration_snapshot` writes a PNG snapshot of the
registered atlas labels -- pass either, both, or neither.

### Core only — whole-graph metrics (`--no_atlas`)

Add `--no_atlas` to skip the atlas layer -- and the FSL dependency it needs -- leaving just the
graph and metric steps:

```bash
python CaravelMetrics.py \
  --image  subject001_MRA.nii.gz \
  --mask   subject001_vessel_mask.nii.gz \
  --output_dir results/ \
  --no_atlas
```

This produces the same metrics, without the per-territory breakdown -- see [Output](#output).
`--image` is still required here (it names the subject's output folder), but no atlas/T1 arguments
are read; if no MRA is available at all, run [the stages individually](#chaining-stages-by-hand) instead.

### All arguments

```
$ python CaravelMetrics.py --help
usage: CaravelMetrics.py [-h] -i IMAGE -m MASK -o OUTPUT_DIR [--t1 T1]
                         [--template TEMPLATE] [--parcellation PARCELLATION]
                         [--label_map LABEL_MAP] [--no_atlas] [--preprocess_mra]
                         [--preprocess_t1] [--mra_stripped] [--t1_stripped]
                         [--min_branch_len MIN_BRANCH_LEN] [--html_qc]
                         [--qc_dir QC_DIR] [--registration_snapshot]

Caravel - Single Subject Processing Pipeline

options:
  -h, --help            show this help message and exit
  -i, --image IMAGE     Path to the subject MRA image.
  -m, --mask MASK       Path to the binary vessel mask.
  -o, --output_dir OUTPUT_DIR
                        Base output directory.
  --t1 T1               Path to the subject T1 image (optional).
  --template TEMPLATE   Path to custom MNI T1 template (optional).
  --parcellation PARCELLATION
                        Path to custom MNI parcellation (optional).
  --label_map LABEL_MAP
                        Path to ArterialAtlasLables.txt for regional naming
                        (optional).
  --no_atlas            Skip atlas registration; graph and metric steps still
                        run, but without regional metrics.
  --preprocess_mra      Apply ANTs denoising + N4 bias correction to the MRA.
  --preprocess_t1       Apply ANTs denoising + N4 bias correction to the T1.
  --mra_stripped        Flag if the input MRA is already skull-stripped.
  --t1_stripped         Flag if the input T1 is already skull-stripped.
  --min_branch_len MIN_BRANCH_LEN
                        Prune terminal branches (leaf-to-bifurcation hairs)
                        shorter than this many mm as skeletonization spurs
                        (default: 2.0).
  --html_qc             Also write an interactive HTML QC report for the
                        extracted graph.
  --qc_dir QC_DIR       Where to write the QC report (default: the subject's
                        own output folder).
  --registration_snapshot
                        Also write a PNG snapshot of the registered atlas
                        labels for QC.
```



## Running individual stages

Each stage can be run on its own. Run them in order (1 → 2 → 3, plus 4 for QC), or skip directly
to the stage you need if its input already exists -- for example, start at stage 3 if you already
have a graph. Every stage is also importable as a Python function (see [Python API](#python-api)).

For each stage below: the **most basic command** first, then the full `--help`.

### Stage 1 — Atlas registration *(optional, brain MRA)*

Warps the arterial atlas into the subject's MRA space. Requires FSL. Useful on its own for
pre-computing atlases for a cohort ahead of a batch graph/metrics run.

```bash
python Atlas_Registration.py \
  --image subject001_MRA.nii.gz \
  --output_dir results/registration/
```

```
$ python Atlas_Registration.py --help
usage: Atlas_Registration.py [-h] --image IMAGE [--t1 T1]
                             [--parcellation PARCELLATION]
                             [--template TEMPLATE] [--output_dir OUTPUT_DIR]
                             [--preprocess_mra] [--preprocess_t1]
                             [--mra_stripped] [--t1_stripped]
                             [--registration_snapshot]

Atlas to MRA Registration Module

options:
  -h, --help            show this help message and exit
  --image IMAGE         Path to the reference MRA image.
  --t1 T1               Path to the subject T1 image.
  --parcellation PARCELLATION
                        Path to custom MNI parcellation.
  --template TEMPLATE   Override path for the MNI template.
  --output_dir OUTPUT_DIR
                        Directory to save outputs.
  --preprocess_mra      Apply ANTs denoising + N4 bias correction to the MRA.
  --preprocess_t1       Apply ANTs denoising + N4 bias correction to the T1.
  --mra_stripped        Flag if the input MRA is already skull-stripped.
  --t1_stripped         Flag if the input T1 is already skull-stripped.
  --registration_snapshot
                        Generate a PNG snapshot of the registered labels for
                        Quality Control.
```

### Stage 2 — Graph extraction

Converts a binary vessel mask into a vessel graph (`.pkl` + `.vtp`). No FSL, no atlas, not
brain-specific -- works for any binary vessel mask (TOF-MRA, CTA, ...).

```bash
python Graph_extractor.py \
  --mask subject001_vessel_mask.nii.gz \
  --output_dir results/
```

```
$ python Graph_extractor.py --help
usage: Graph_extractor.py [-h] --mask MASK --output_dir OUTPUT_DIR
                          [--smooth_iters SMOOTH_ITERS]
                          [--smooth_alpha SMOOTH_ALPHA]
                          [--smooth_reference_spacing SMOOTH_REFERENCE_SPACING]
                          [--intensity_thresh INTENSITY_THRESH]
                          [--orphan_dist ORPHAN_DIST] [--merge_dist MERGE_DIST]
                          [--target_iso TARGET_ISO]
                          [--min_branch_len MIN_BRANCH_LEN]

Vessel Graph Extractor Module

options:
  -h, --help            show this help message and exit
  --mask MASK           Path to the binary vessel mask NIfTI file.
  --output_dir OUTPUT_DIR
                        Directory to save the resulting .pkl and .vtp files.
  --smooth_iters SMOOTH_ITERS
                        Laplacian smoothing iterations at the reference
                        resolution (default: 2); scaled per-scan by voxel
                        spacing.
  --smooth_alpha SMOOTH_ALPHA
                        Laplacian smoothing alpha (default: 0.8).
  --smooth_reference_spacing SMOOTH_REFERENCE_SPACING
                        Reference isotropic spacing (mm) that --smooth_iters is
                        calibrated for (default: 0.5).
  --intensity_thresh INTENSITY_THRESH
                        Intensity threshold for edge pruning (default: 0.6).
  --orphan_dist ORPHAN_DIST
                        Orphan detection distance threshold mm (default: 2.5).
  --merge_dist MERGE_DIST
                        Orphan merge distance threshold mm (default: 3.0).
  --target_iso TARGET_ISO
                        Target isotropic voxel spacing in mm (default: auto --
                        this image's finest native axis, so no axis is
                        downsampled).
  --min_branch_len MIN_BRANCH_LEN
                        Prune terminal branches (leaf-to-bifurcation hairs)
                        shorter than this many mm as skeletonization spurs
                        (default: 2.0); raise to remove more hairs.
```

### Stage 3 — Metric computation

Computes morphometrics from a graph. Pass `--atlas` (a registered atlas from stage 1) for the
regional breakdown; pass `--mask` for exact voxel-counted volume. Both optional -- with neither,
you get whole-graph metrics only.

```bash
python Metric_extractor.py \
  --graph results/subject001_vessel_mask_vessel_data.pkl \
  --out_dir results/
```

```
$ python Metric_extractor.py --help
usage: Metric_extractor.py [-h] --graph GRAPH --out_dir OUT_DIR [--mask MASK]
                           [--label_map LABEL_MAP] [--atlas ATLAS]

Extract Metrics from Saved Vessel Graph

options:
  -h, --help            show this help message and exit
  --graph GRAPH         Path to the .pkl graph file.
  --out_dir OUT_DIR     Directory to save the CSV metrics files.
  --mask MASK           Path to original binary mask for EXACT volume
                        calculation (optional).
  --label_map LABEL_MAP
                        Path to ArterialAtlasLables.txt (optional).
  --atlas ATLAS         Path to the registered atlas NIfTI (optional).
```

### Stage 4 — QC report *(optional)*

Writes a self-contained interactive HTML report per subject: a 3D view of the skeleton graph, the
segmentation surface (with `--mask`), and sortable per-segment stats -- computed with the *same*
functions as stage 3, so the numbers match the CSVs. Point it at one graph (`--graph`) or a whole
results tree (`--results_dir`, batch mode).

```bash
python QC_report.py \
  --graph results/subject001_vessel_mask_vessel_data.pkl \
  --out results/QC/
```

```
$ python QC_report.py --help
usage: QC_report.py [-h] (--graph GRAPH | --results_dir RESULTS_DIR) --out OUT
                    [--mask MASK] [--atlas ATLAS] [--mask_dir MASK_DIR]
                    [--atlas_dir ATLAS_DIR] [--label_map LABEL_MAP]
                    [--plotlyjs {directory,inline,cdn}]
                    [--mesh_faces MESH_FACES] [--no_mesh]
                    [--max_segments_per_root MAX_SEGMENTS_PER_ROOT]
                    [--skip_existing] [--line_width LINE_WIDTH]

Generate interactive HTML QC reports for Caravel vessel graphs.

options:
  -h, --help            show this help message and exit
  --graph GRAPH         Path to a single <subject>_vessel_data.pkl.
  --results_dir RESULTS_DIR
                        Directory tree to scan for *_vessel_data.pkl.
  --out OUT             Output directory for the HTML reports.
  --mask MASK           Binary vessel mask NIfTI for the translucent surface
                        and the voxel-counted volume (single-graph mode).
  --atlas ATLAS         Registered arterial atlas NIfTI for region colouring
                        (single-graph mode).
  --mask_dir MASK_DIR   Directory of masks, matched to subjects by name (batch
                        mode).
  --atlas_dir ATLAS_DIR
                        Directory of registered atlases, matched by name (batch
                        mode).
  --label_map LABEL_MAP
                        ArterialAtlasLables.txt for region names.
  --plotlyjs {directory,inline,cdn}
                        How to ship plotly.js: 'directory' (default) writes one
                        shared file next to the reports; 'inline' embeds it in
                        each file (~4.9 MB); 'cdn' needs internet.
  --mesh_faces MESH_FACES
                        Target triangle count for the decimated mask surface
                        (default: 45000).
  --no_mesh             Skip the mask surface even when a mask is available.
  --max_segments_per_root MAX_SEGMENTS_PER_ROOT
                        Cap segments stored per root, keeping the longest
                        (default: all).
  --skip_existing       Do not regenerate reports that already exist.
  --line_width LINE_WIDTH
                        Initial skeleton line width shown in the report;
                        adjustable afterwards with the in-page slider (default:
                        3.0).
```

### Chaining stages by hand

Chaining stages 2 → 3 is exactly what `--no_atlas` does internally, minus registration. Use this
when no MRA/reference image is available, or when working from a mask produced elsewhere:

```bash
python Graph_extractor.py \
  --mask subject001_vessel_mask.nii.gz \
  --output_dir results/

python Metric_extractor.py \
  --graph results/subject001_vessel_mask_vessel_data.pkl \
  --out_dir results/ \
  --mask subject001_vessel_mask.nii.gz
```

### Already have a graph

Graph extraction can be skipped as well. `Metric_extractor` reads more than Caravel's own pickle:
`robust_graph_loader` auto-detects a raw NetworkX graph, a UNC Skwerer/Bullitt tree dictionary, or
a MetaIO/TubeTK `.tre` scene, so any of these work as `--graph` directly. This supports structures
from other vessel-extraction tools (e.g. a coronary artery tree exported as `.tre`):

```bash
python Metric_extractor.py --graph my_vessel_tree.pkl --out_dir results/
python Metric_extractor.py --graph my_vessel_tree.tre --out_dir results/
```

### Python API

Every stage is a plain function, so the pipeline can be embedded directly in your own scripts.

```python
from Atlas_Registration import process_registration
from Graph_extractor import extract_graph
from Metric_extractor import extract_metrics

registered_atlas_path, inv_matrix_path = process_registration(
    image_path="subject001_MRA.nii.gz",
    output_dir="results/registration/",
    image_t1_path="subject001_T1.nii.gz",
    mra_stripped=True,
    t1_stripped=True,
)

graph_pkl_path = extract_graph(
    nifti_path="subject001_vessel_mask.nii.gz",
    output_folder="results/",
)

extract_metrics(
    output_folder="results/",
    graph_pkl_path=graph_pkl_path,
    mask_path="subject001_vessel_mask.nii.gz",     # optional: exact volume
    registered_atlas_path=registered_atlas_path,   # optional: regional metrics
)
```

`extract_metrics` always computes the full metric set; there is no metric-selection argument. Omit
`registered_atlas_path` for whole-graph-only metrics, and `mask_path` when the exact voxel-counted
volume is not required.



## Output

**With the atlas layer**, `results/subject001_MRA/` contains the registered/dilated atlas, the vessel graph
(`.pkl` + `.vtp`), `subject001_MRA_global_summary.csv` (whole-graph metrics),
`subject001_MRA_region_summary.csv` + `subject001_MRA_all_components_by_region.csv`
(per-territory), plus the two independent QC outputs (`--html_qc`, `--registration_snapshot`) --
every file is prefixed with the subject name, so files do not collide when several subjects'
folders are gathered into one place:

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
    ├── subject001_MRA_QC.html                                # HTML graph report (with --html_qc)
    ├── subject001_MRA_QC_snapshot.png                        # registration PNG (with --registration_snapshot)
    ├── regional_vtps/                                        # one VTP per atlas territory
    ├── regional_results/                                     # per-region, per-component detail + segment CSVs
    └── processing.log
```

The HTML report defaults into the subject's own folder alongside everything else; pass `--qc_dir`
to put it somewhere else instead (e.g. a shared `QC/` folder across subjects for a batch run). The
PNG snapshot is always written next to the registration outputs, since it's generated as part of
that step.

**Core only** (`--no_atlas`, or the manual `Graph_extractor.py`/`Metric_extractor.py` chain)
produces no per-territory breakdown and no `global_summary.csv`. Metrics are written directly into
`output_folder` as `region_summary.csv` (with a single `global` pseudo-region) +
`all_components_by_region.csv`, both still prefixed with the subject name. This is
`extract_metrics`'s convention whenever `registered_atlas_path` is `None`, so it applies whether the
run used `CaravelMetrics.py --no_atlas` or the individual stages. `--registration_snapshot` has no
effect with `--no_atlas`, as there is no registration step to snapshot.



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
