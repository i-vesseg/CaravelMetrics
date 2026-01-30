## 💻 Usage

CaravelMetrics is implemented as a batch processing pipeline that processes vessel segmentation masks and extracts features across multiple subjects.

### Mode Selection

Choose your analysis mode based on your needs:

| Mode | Files Needed | FSL Required | Use Case |
|------|--------------|--------------|----------|
| **Whole-Brain** | Image folder + Mask folder | ❌ No | Quick screening, global metrics |
| **Atlas-Based** | Image folder + Mask folder + T1 folder + Atlas | ✅ Yes | Regional analysis, territory comparison |

### Configuration

CaravelMetrics uses a configuration-based approach. Edit the following variables in `CaravelMetrics.py`:

```python
# Set paths to your data folders
segmentation_folder = 'path/to/vessel_masks'  # Folder containing vessel segmentation masks
image_folder = 'path/to/vessel_images'        # Folder containing vessel images (e.g., TOF-MRA)
t1_image_folder = 'path/to/T1_images'         # Folder containing T1 images (atlas mode only)

# Atlas configuration (for regional analysis)
atlas_path = 'ArterialAtlas.nii.gz'           # Path to arterial atlas file
label_map_path = 'ArterialAtlas_labels.txt'   # Path to atlas label definitions

# Output configuration
output_base_folder = 'Results/'                # Base folder for all results

# Analysis mode
use_atlas = False  # Set to True for atlas-based regional analysis, False for whole-brain

# Processing configuration
skip_existing = False     # Set to True to skip already processed files
run_parallel = True       # Set to True to use parallel processing
num_workers = None        # Number of parallel workers (None = auto-detect)
chunk_size = 1           # Chunk size for parallel processing

# Metric selection
selected_metrics = ['total_length', 'num_bifurcations', 'bifurcation_density', 
                   'volume', 'fractal_dimension', 'lacunarity', ...]
```

### Running the Pipeline

#### 1. Configure Your Settings

Edit `CaravelMetrics.py` to set your data paths and analysis parameters:

```python
# Example configuration for whole-brain analysis
segmentation_folder = '/data/vessel_segmentations'
image_folder = '/data/TOF_MRA_images'
output_base_folder = 'Results_WholeBrain/'
use_atlas = False
run_parallel = True
```

#### 2. Run the Pipeline

```bash
python CaravelMetrics.py
```

The pipeline will automatically:
- Discover all `.nii`, `.nii.gz` files in the segmentation folder
- Process each file (optionally in parallel)
- Generate results in the output folder
- Create a log file (`processing.log`) with detailed execution information

### Example Configurations

#### 1. Whole-Brain Analysis (Fast Screening)

```python
# CaravelMetrics.py configuration
segmentation_folder = 'data/segmentations/'
image_folder = 'data/TOF_images/'
output_base_folder = 'Results_WholeBrain/'

use_atlas = False  # Whole-brain mode
run_parallel = True
num_workers = 4
skip_existing = False

selected_metrics = ['total_length', 'volume', 'bifurcation_density', 
                   'fractal_dimension', 'num_loops']
```

#### 2. Atlas-Based Regional Analysis (Detailed)

```python
# CaravelMetrics.py configuration
segmentation_folder = 'data/segmentations/'
image_folder = 'data/TOF_images/'
t1_image_folder = 'data/T1_images/'

atlas_path = 'atlases/ArterialAtlas.nii.gz'
label_map_path = 'atlases/ArterialAtlas_labels.txt'
output_base_folder = 'Results_Regional/'

use_atlas = True  # Atlas-based regional mode
run_parallel = True
num_workers = 2  # Use fewer workers for atlas mode (more memory intensive)
skip_existing = False

selected_metrics = ['total_length', 'spline_mean_curvature', 
                   'fractal_dimension', 'bifurcation_density']
```

#### 3. Resume Interrupted Processing

```python
# Skip files that have already been processed
skip_existing = True  # Set to True
run_parallel = True
# Processed files are skipped automatically
```

#### 4. Sequential Processing for Debugging

```python
# Disable parallel processing to see detailed error messages
run_parallel = False  # Sequential processing
skip_existing = False
```

### Advanced Usage: Direct Script Access

For single-subject processing or custom workflows, you can use the underlying scripts directly:

```python
from Atlas_Registration import process_registration
from Graph_extractor import extract_graph
from Metric_extractor import extract_metrics

# Step 1: Register atlas (atlas mode only)
process_registration(
    image_path='sub001_TOF.nii.gz',
    t1_image_path='sub001_T1.nii.gz',
    mask_path='sub001_mask.nii.gz',
    atlas_path='ArterialAtlas.nii.gz',
    output_folder='output/sub001/'
)

# Step 2: Extract vessel graph
extract_graph(
    nifti_path='sub001_mask.nii.gz',
    output_folder='output/sub001/'
)

# Step 3: Compute metrics
extract_metrics(
    patient_name='sub001',
    output_folder='output/sub001/',
    graph_pkl_path='output/sub001/vessel_data.pkl',
    selected_metrics=['total_length', 'volume', 'fractal_dimension'],
    label_map_path='ArterialAtlas_labels.txt',
    registered_atlas_path='output/sub001/sub001_T1_registered_atlas.nii.gz'  # None for whole-brain
)
```

---

## 📁 Input Data Requirements

### Data Organization

Organize your data into separate folders:

```
project/
├── segmentations/          # Vessel segmentation masks
│   ├── sub001_MRA.nii.gz
│   ├── sub002_MRA.nii.gz
│   └── ...
├── TOF_images/             # Original vessel images
│   ├── sub001_MRA.nii.gz
│   ├── sub002_MRA.nii.gz
│   └── ...
├── T1_images/              # T1-weighted images (atlas mode only)
│   ├── sub001_T1.nii.gz
│   ├── sub002_T1.nii.gz
│   └── ...
└── atlases/
    ├── ArterialAtlas.nii.gz
    └── ArterialAtlas_labels.txt
```

**Important**: File names must be consistent across folders. The pipeline matches files by name (after removing extensions).

### Vessel Segmentation Masks
- **Format**: NIfTI (.nii or .nii.gz)
- **Type**: Binary masks (1 = vessel, 0 = background)
- **Source**: TOF-MRA, CTA, or other angiographic modalities
- **Preprocessing**: Should be skull-stripped

### Vessel Images
- **Format**: NIfTI (.nii or .nii.gz)
- **Type**: Raw TOF-MRA or CTA images
- **Naming**: Must match corresponding mask files

### T1-Weighted Images (Atlas-Based Mode Only)
- **Format**: NIfTI (.nii or .nii.gz)
- **Purpose**: Intermediate registration step for atlas alignment
- **Requirements**: Must correspond to the same subject as the vessel image
- **Naming**: File name should contain 'T1' and subject ID

### Arterial Atlas (Atlas-Based Mode Only)
- **Format**: NIfTI (.nii or .nii.gz)
- **Type**: Integer-labeled regions (1-30 for 30 territories)
- **Recommended**: Liu et al. 2023 Digital 3D Brain MRI Arterial Territories Atlas
- **Label Map**: Text file with format: `<label_id> <region_name>`
- **Note**: Atlas labels must be integers; label 0 is treated as background

---

## 📊 Output Structure

After processing, the output folder contains:

### Whole-Brain Mode Output
```
Results/
├── sub001/
│   ├── vessel_data.pkl                    # Serialized graph structure
│   ├── vessel_graph.vtp                   # VTK visualization file
│   ├── Region_1/                          # Whole-brain results
│   │   ├── Conn_comp_1/                   # Largest connected component
│   │   │   ├── Conn_comp_1_skeleton.nii.gz
│   │   │   └── Segments/                  # Vessel segments (optional)
│   │   ├── Conn_comp_2/                   # Second largest component
│   │   └── ...
│   └── processing.log                     # Execution log
├── sub001_GLOBAL/
│   ├── all_components_by_region.csv       # Detailed component metrics
│   └── region_summary.csv                 # Aggregated metrics
└── sub002/
    └── ...
```

### Atlas-Based Mode Output
```
Results/
├── sub001/
│   ├── vessel_data.pkl                    # Serialized graph structure
│   ├── vessel_graph.vtp                   # VTK visualization file
│   ├── sub001_T1_registered_atlas.nii.gz  # Atlas registered to TOF space
│   ├── sub001_T1_brain.nii.gz             # Skull-stripped T1
│   ├── sub001_T1_cropped.nii.gz           # Cropped T1
│   ├── sub001_T1_in_MRA.nii.gz            # T1 registered to MRA space
│   ├── Region_1/                          # First atlas region
│   │   ├── Conn_comp_1/
│   │   │   ├── Conn_comp_1_skeleton.nii.gz
│   │   │   └── Segments/
│   │   ├── Conn_comp_2/
│   │   └── ...
│   ├── Region_2/                          # Second atlas region
│   │   └── ...
│   ├── all_components_by_region.csv       # Detailed component metrics per region
│   ├── region_summary.csv                 # Aggregated metrics per region
│   └── processing.log                     # Execution log
└── sub002/
    └── ...
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
