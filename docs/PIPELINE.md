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
1. **Mesh Generation**: Create surface mesh from vessel mask for geodesic distance computation
2. **Skeletonization**: Apply 3D morphological thinning to reduce vessel mask to centerlines
3. **Node Detection**: Identify branch points using 26-connectivity neighborhood analysis:
   - Endpoints (degree = 1)
   - Bifurcations (degree = 3)
   - Abnormal nodes (degree > 3, potential artifacts)
4. **Graph Construction**: Build NetworkX graph with:
   - Nodes: Branch points with 3D spatial coordinates
   - Edges: Vessel segments with Euclidean distance weights
   - Node attributes: Position, degree, multiplicity
5. **Orphan Node Handling**: Connect isolated branches within distance threshold
6. **Triangle Removal**: Clean discretization artifacts by removing spurious 3-node loops
7. **Laplacian Smoothing**: Apply iterative smoothing to node positions
8. **Geodesic Distance**: Compute mesh-based geodesic distances for tortuosity analysis
9. **Output**: Serialized graph structure (PKL) and VTK visualization file

#### Module 3: Metric Extraction (`Metric_extractor.py`)
**Executed for all modes**
1. **Input Loading**: Load graph structure and optional registered atlas
2. **Regional Segmentation** (atlas mode only): Partition graph nodes by atlas labels
3. **Connected Component Analysis**: Identify separate vessel networks per region
4. **Component Ranking**: Sort components by size (number of edges)
5. **Segment Extraction**: For each component:
   - Find all endpoints
   - Compute shortest paths between endpoint pairs
   - Extract individual vessel segments with path multiplicity tracking
6. **Metric Computation**: For each selected metric:
   - **Morphometric**: Total length, volume, diameter
   - **Topological**: Bifurcation count/density, loops, abnormal nodes
   - **Fractal**: Box-counting dimension, lacunarity
   - **Geometric**: Arc-length parameterized tortuosity metrics with multiplicity weighting
7. **Aggregation**: Combine metrics across components within each region
8. **Output**: CSV files with component-level and region-level metrics

### Batch Processing Workflow

The main `CaravelMetrics.py` script orchestrates the three modules:

1. **Discovery**: Scan segmentation folder for all `.nii`/`.nii.gz` files
2. **File Matching**: Match vessel masks with corresponding images and T1 scans
3. **Parallel Execution**: Process multiple subjects simultaneously (configurable workers)
4. **Progress Tracking**: Display progress bars and log detailed execution information
5. **Error Handling**: Gracefully handle failures, continue processing remaining subjects
6. **Summary Report**: Generate success/failure statistics at completion

---
