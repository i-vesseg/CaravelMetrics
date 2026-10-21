## 🔧 Pipeline Overview

<!-- Pipeline Diagram -->
<p align="center">
  <img src="../logos/pipeline_pp.png" alt="CaravelMetrics Pipeline" width="800"/>
</p>

### Analysis Modes

CaravelMetrics offers two complementary analysis modes:

#### 1. **Whole-Brain Analysis Mode** (Fast & Simple)
- Analyzes entire vessel network as a single region
- **Requirements**: Only vessel image and mask
- **Use case**: Exploratory analysis, rapid screening, global metrics

#### 2. **Atlas-Based Regional Analysis Mode** (Detailed)
- Parcellates vessels into anatomical territories
- **Requirements**: Vessel image, mask, T1 image, and arterial atlas
- **Use case**: Regional comparisons, territory-specific analysis, population studies

### Processing Steps

The pipeline consists of three main modules that can run independently or as part of the automated batch workflow:

#### Module 1: Atlas Registration (`Atlas_Registration.py`)
**Only executed in atlas-based mode**
1. **T1 Preprocessing**: 
   - Crop T1 image (remove neck region)
   - Extract brain using FSL BET (frac=0.30)
   - Reorient to standard orientation
2. **MRA Preprocessing**:
   - Reorient TOF-MRA to standard orientation
3. **Multi-Step Registration**:
   - Register T1 to MRA space using FLIRT (12 DOF, mutual information)
   - Apply brain mask to atlas
   - Register masked atlas to T1 using FLIRT
   - Register full atlas using masked registration as initialization
   - Apply combined transformation to align atlas with TOF-MRA space
4. **Output**: Registered atlas in TOF-MRA space

#### Module 2: Graph Extraction (`Graph_extractor.py`)
**Executed for all modes**
1. **Load and crop**: binarise the mask and crop it to its bounding box (+5 mm margin)
2. **Fill holes**: fill internal cavities (`binary_fill_holes`)
3. **Isotropic resampling**: resample to the finest native axis, or to `--target_iso` (signed-distance interpolation)
4. **Optional boundary smoothing** (`--mask_smoothing`, off by default; for noisy segmentations only; sigma at least 0.7 voxel)
5. **Surface mesh**: isosurface used by the edge check below
6. **Skeletonisation**: 3D thinning (scikit-image); nodes = skeleton voxels, edges = 26-neighbours
7. **Triangle collapse**: remove the longest edge of every 3-node clique (junction clean-up)
8. **Edge check**: an edge whose straight line leaves the mask is cut if the path along the vessel surface between its ends is more than twice the edge length (separates touching vessels)
9. **Small components**: drop components shorter than 3.5 mm
10. **Terminal pruning**: remove leaf-to-junction branches shorter than `--min_branch_len` (2 mm)
11. **Tip extension**: extend skeleton tips that stop short of their vessel end (on by default; `--no_extend_tips`)
12. **Rung contraction**: merge junctions less than 1 mm apart
13. **Laplacian smoothing** of node positions (dose scaled with voxel size)
14. **Radius**: equivalent-cylinder radius from the local mask volume (EDT value kept as `radius_edt`)
15. **Output**: graph (`.pkl`, with parameters, code version and QC), VTP files, `*_graph_qc.json`

#### Module 3: Metric Extraction (`Metric_extractor.py`, `Branch_metrics.py`)
**Executed for all modes**
1. **Input loading**: Caravel `.pkl`, raw NetworkX graph, UNC tree dictionary or TubeTK `.tre`
2. **Regional split** (atlas mode only): each node goes to the atlas region it lies in (vessels are cut at region borders)
3. **Per connected component**: length, volume, bifurcations, loops, segments, fractal dimension, lacunarity
4. **Per segment**: length, arc/chord, diameter, smoothing-spline curvature
5. **Per junction**: branching angles; junctions whose daughters rejoin within a short loop are flagged and left out of the median
6. **Aggregation**: sums, length-weighted means, and medians pooled over segments/junctions, per region and for the whole network; FD and lacunarity on each region's whole network
7. **Output**: component, region, segment and bifurcation CSVs, a QC table with flags, and a run-info file

### Processing several subjects

`CaravelMetrics.py` processes one subject per call (registration, graph
extraction, metrics, optional QC report). For a cohort, loop over subjects,
e.g. with masks named `<subject>_mask.nii.gz` next to `<subject>_MRA.nii.gz`:

```bash
for m in data/*_mask.nii.gz; do
  s=$(basename "$m" _mask.nii.gz)
  python CaravelMetrics.py -i "data/${s}_MRA.nii.gz" -m "$m" -o "results/$s" \
    || echo "$s failed" >> results/failed.txt
done
```

Each subject's folder then holds its summary, segment, bifurcation and QC
tables (`*_region_summary.csv`, `*_segments.csv`, `*_bifurcations.csv`,
`*_qc.csv`) and `*_run_info.json` with the code version and settings; for a
cohort table concatenate the `*_region_summary.csv` files. Use one
`--target_iso` for the whole cohort when scans differ in resolution.

---
