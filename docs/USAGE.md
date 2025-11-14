## 💻 Usage

CaravelMetrics is implemented as a command-line tool that processes vessel segmentation masks and extracts features.

### Mode Selection

Choose your analysis mode based on your needs:

| Mode | Files Needed | FSL Required | Use Case |
|------|--------------|--------------|----------|
| **Whole-Brain** | 2 (image + mask) | ❌ No | Quick screening, global metrics |
| **Atlas-Based** | 4 (image + mask + T1 + atlas) | ✅ Yes | Regional analysis, territory comparison |

### Basic Command Structure

**Whole-Brain Mode:**
```bash
python CaravelMetrics.py <image_path> <mask_path> [OPTIONS]
```

**Atlas-Based Mode:**
```bash
python CaravelMetrics.py <image_path> <mask_path> --image_T1 <T1_path> --atlas_path <atlas_path> [OPTIONS]
```

### Required Arguments

**For Both Modes:**
- `image_path`: Path to the vessel image (.nii or .nii.gz) - typically TOF-MRA
- `mask_path`: Path to the binary vessel segmentation mask (.nii or .nii.gz)

**Additional for Atlas-Based Mode:**
- `--image_T1`: Path to the T1-weighted image (.nii or .nii.gz) - used for atlas registration
- `--atlas_path`: Path to the arterial atlas file (.nii or .nii.gz) for regional parcellation

### Optional Arguments

- `--metrics METRIC [METRIC ...]`: Specific metrics to compute (default: all metrics)
  - Available metrics: `total_length`, `num_bifurcations`, `bifurcation_density`, `volume`, `fractal_dimension`, `lacunarity`, `geodesic_length`, `avg_diameter`, `spline_arc_length`, `spline_chord_length`, `spline_mean_curvature`, `spline_mean_square_curvature`, `spline_rms_curvature`, `arc_over_chord`, `fit_rmse`, `num_loops`, `num_abnormal_degree_nodes`
  
- `--output_folder PATH`: Directory to save results (default: `./VESSEL_METRICS`)

- `--no_segment_masks`: Disable saving individual vessel segment masks

- `--no_conn_comp_masks`: Disable saving connected component reconstructions

### Example Usage

#### 1. Whole-Brain Analysis (Fast Screening)

```bash
# Basic whole-brain analysis
python CaravelMetrics.py \
    data/sub001_TOF.nii.gz \
    data/sub001_vessels.nii.gz \
    --output_folder results/sub001/

# With specific metrics only
python CaravelMetrics.py \
    data/sub001_TOF.nii.gz \
    data/sub001_vessels.nii.gz \
    --metrics total_length volume bifurcation_density \
    --output_folder results/sub001/
```

#### 2. Atlas-Based Regional Analysis (Detailed)

```bash
# Full regional analysis with atlas
python CaravelMetrics.py \
    data/sub001_TOF.nii.gz \
    data/sub001_vessels.nii.gz \
    --image_T1 data/sub001_T1.nii.gz \
    --atlas_path atlases/arterial_atlas.nii.gz \
    --output_folder results/sub001/

# With specific metrics
python CaravelMetrics.py \
    data/sub001_TOF.nii.gz \
    data/sub001_vessels.nii.gz \
    --image_T1 data/sub001_T1.nii.gz \
    --atlas_path atlases/arterial_atlas.nii.gz \
    --metrics total_length spline_mean_curvature fractal_dimension \
    --output_folder results/sub001/
```

#### 3. Process Without Saving Intermediate Masks

```bash
# Whole-brain mode
python CaravelMetrics.py \
    data/sub001_TOF.nii.gz \
    data/sub001_vessels.nii.gz \
    --no_segment_masks \
    --no_conn_comp_masks \
    --output_folder results/sub001/

# Atlas-based mode
python CaravelMetrics.py \
    data/sub001_TOF.nii.gz \
    data/sub001_vessels.nii.gz \
    --image_T1 data/sub001_T1.nii.gz \
    --atlas_path atlases/arterial_atlas.nii.gz \
    --no_segment_masks \
    --no_conn_comp_masks \
    --output_folder results/sub001/
```

#### 4. Batch Processing Multiple Subjects

```bash
#!/bin/bash
# batch_process.sh - Whole-brain mode

OUTPUT_BASE="results"

for subject in sub001 sub002 sub003; do
    echo "Processing ${subject} (whole-brain)..."
    python CaravelMetrics.py \
        data/${subject}_TOF.nii.gz \
        data/${subject}_vessels.nii.gz \
        --output_folder ${OUTPUT_BASE}/${subject}/ \
        --no_segment_masks \
        --no_conn_comp_masks
done

echo "Batch processing complete!"
```

```bash
#!/bin/bash
# batch_process_regional.sh - Atlas-based mode

ATLAS="atlases/arterial_atlas.nii.gz"
OUTPUT_BASE="results"

for subject in sub001 sub002 sub003; do
    echo "Processing ${subject} (atlas-based)..."
    python CaravelMetrics.py \
        data/${subject}_TOF.nii.gz \
        data/${subject}_vessels.nii.gz \
        --image_T1 data/${subject}_T1.nii.gz \
        --atlas_path ${ATLAS} \
        --output_folder ${OUTPUT_BASE}/${subject}/ \
        --no_segment_masks \
        --no_conn_comp_masks
done

echo "Batch processing complete!"
```

#### 5. Hybrid Workflow (Screening + Detailed Analysis)

```bash
#!/bin/bash
# Two-stage analysis: Fast screening followed by detailed regional analysis

# Stage 1: Screen all subjects with whole-brain mode (~15 min each)
for subject in sub001 sub002 sub003 sub004 sub005; do
    echo "Screening ${subject}..."
    python CaravelMetrics.py \
        data/${subject}_TOF.nii.gz \
        data/${subject}_vessels.nii.gz \
        --output_folder results/screening/${subject}/ \
        --no_segment_masks --no_conn_comp_masks
done

# Stage 2: Detailed regional analysis for selected subjects (~60 min each)
SELECTED="sub001 sub003 sub005"
ATLAS="atlases/arterial_atlas.nii.gz"

for subject in ${SELECTED}; do
    echo "Detailed analysis of ${subject}..."
    python CaravelMetrics.py \
        data/${subject}_TOF.nii.gz \
        data/${subject}_vessels.nii.gz \
        --image_T1 data/${subject}_T1.nii.gz \
        --atlas_path ${ATLAS} \
        --output_folder results/detailed/${subject}/
done
```

---

## 📁 Input Data Requirements

### Vessel Segmentation Masks
- **Format**: NIfTI (.nii or .nii.gz)
- **Type**: Binary masks (1 = vessel, 0 = background)
- **Source**: TOF-MRA, CTA, or other angiographic modalities
- **Preprocessing**: Should be skull-stripped

### T1-Weighted Images (Atlas-Based Mode Only)
- **Format**: NIfTI (.nii or .nii.gz)
- **Purpose**: Intermediate registration step for atlas alignment
- **Requirements**: Should correspond to the same subject as the vessel image

### Arterial Atlas (Atlas-Based Mode Only)
- **Format**: NIfTI (.nii or .nii.gz)
- **Type**: Integer-labeled regions (1-30 for 30 territories)
- **Recommended**: Liu et al. 2023 Digital 3D Brain MRI Arterial Territories Atlas
- **Note**: Atlas labels must be integers; label 0 is treated as background

---

## 📊 Output Structure

After processing, the output folder contains:

### Whole-Brain Mode Output
```
results/sub001/
├── Region_1/                          # Whole-brain results
│   ├── Conn_comp_1/                   # Largest connected component
│   │   ├── Conn_comp_1_skeleton.nii.gz
│   │   └── Segments/                  # Vessel segments (optional)
│   ├── Conn_comp_2/                   # Second largest component
│   └── ...
├── all_components_by_region.csv       # Detailed component metrics
└── region_summary.csv                 # Aggregated metrics
```

### Atlas-Based Mode Output
```
results/sub001/
├── Region_1/                          # First atlas region
│   ├── Conn_comp_1/
│   │   ├── Conn_comp_1_skeleton.nii.gz
│   │   └── Segments/
│   ├── Conn_comp_2/
│   └── ...
├── Region_2/                          # Second atlas region
│   └── ...
├── Region_N/                          # Nth atlas region
│   └── ...
├── sub001_registered_atlas.nii.gz    # Atlas registered to TOF space
├── all_components_by_region.csv      # Detailed component metrics per region
└── region_summary.csv                # Aggregated metrics per region
```

### CSV Output Format

**all_components_by_region.csv** - Detailed metrics for each connected component:
```csv
region_label,total_length,volume,num_bifurcations,bifurcation_density,fractal_dimension,lacunarity,...
1,2543.21,1234.56,45,0.0177,1.67,0.42,...
1,876.43,342.12,12,0.0137,1.54,0.38,...
2,1543.87,892.34,32,0.0208,1.71,0.45,...
...
```

**region_summary.csv** - Aggregated metrics per region:
```csv
region_label,num_components,total_length,volume,num_bifurcations,...
1,15,45234.5,12456.8,234,...
2,12,38765.2,10234.5,189,...
...
```

**Note**: In whole-brain mode, all results are aggregated under `region_label = 1`.

---
