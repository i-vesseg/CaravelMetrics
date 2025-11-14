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

**Whole-Brain Mode:**
1. **Input Loading**: Load vessel image and mask
2. **Skeletonization**: Reduce vessel mask to centerline representation
3. **Graph Construction**: Build mathematical graph (nodes = branch points, edges = segments)
4. **Spurious Loop Removal**: Clean discretization artifacts
5. **Feature Extraction**: Compute all requested metrics for entire network
6. **Results Export**: Save CSV files and optional visualization masks

**Atlas-Based Mode:**
1. **Input Loading**: Load TOF-MRA, T1, vessel mask, and arterial atlas
2. **Brain Extraction**: Apply FSL BET to T1 image
3. **Reorientation**: Standardize orientation to MNI space
4. **Atlas Registration**: 
   - Register atlas to T1 using FLIRT (mutual information, 12 DOF)
   - Register T1 to TOF-MRA using FLIRT
   - Apply combined transformation to align atlas with TOF-MRA
5. **Regional Segmentation**: Partition vessels according to atlas regions
6. **Skeletonization**: Reduce vessel masks to centerline representations
7. **Graph Construction**: Build mathematical graphs per region
8. **Spurious Loop Removal**: Clean discretization artifacts
9. **Feature Extraction**: Compute all requested metrics per region
10. **Results Export**: Save CSV files and optional visualization masks

---
