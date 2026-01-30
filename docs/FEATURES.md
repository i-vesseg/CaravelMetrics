## 🔬 Features

CaravelMetrics extracts 15 features organized into four categories:

### Morphometric Features
- **Total Vessel Length**: Structural integrity of vascular network
- **Volume**: Total vascular volume from radius estimates
- **Average Diameter**: Mean vessel diameter across the network

### Topological Features
- **Bifurcation Count & Density**: Branching complexity and organization
- **Number of Loops**: Collateral circulation paths
- **Abnormal Degree Nodes**: Detection of anomalous branching patterns (nodes with degree > 3)
- **Connected Components**: Network integrity and fragmentation analysis

### Fractal Features
- **Fractal Dimension**: Multi-scale self-similar organization via box-counting
- **Lacunarity**: Spatial heterogeneity and distribution patterns

### Geometric Features (Tortuosity Metrics)
- **Geodesic Length**: Arc-length parameterized path measurements
- **Spline Arc Length & Chord Length**: Path length measurements via B-spline fitting
- **Mean Curvature**: Average tortuosity weighted by path multiplicity
- **Mean Square Curvature**: Curvature variability along vessels
- **RMS Curvature**: Root mean square curvature
- **Arc-over-Chord Ratio**: Global segment tortuosity metric
- **Fit RMSE**: Spline fitting quality metric

---

## 📊 Validation Results

Applied to the IXI dataset (570 TOF-MRA volumes, ages 20-86):

### Age-Related Changes
- **20% decline** in total vessel length (r = -0.50, p < 0.001, η² = 0.204)
- **Reduced bifurcations** (η² = 0.064), indicating network simplification
- **Increased tortuosity** (r ≈ +0.10), consistent with arterial stiffening
- Preserved fractal organization despite structural changes

### Demographic Associations
- **Sex differences**: Males show higher bifurcation density, loops, and volume
- **BMI effects**: Normal-weight individuals demonstrate longer vessels (η² = 0.032)
- **Education gradient**: Stepwise increase in vessel length from no qualification to university degree (η² = 0.055)
- **Height correlations**: Positive associations with network complexity (η² = 0.026-0.053)

---

## 📖 Feature Definitions

### Morphometric Features

- **Total Length**: Sum of Euclidean distances between all adjacent graph nodes (mm)
- **Volume**: Computed from local radius estimates via distance transform (mm³)
- **Average Diameter**: Mean vessel diameter across all centerline points (mm)

### Topological Features

- **Number of Bifurcations**: Count of nodes with degree = 3
- **Bifurcation Density**: Bifurcations per unit length (bifurcations/mm)
- **Number of Loops**: Count of cycles in the vessel graph (collateral circulation)
- **Number of Abnormal Degree Nodes**: Nodes with degree > 3 (potential artifacts)

### Fractal Features

- **Fractal Dimension**: Slope of log(N(ε)) vs log(1/ε) from box-counting method
  - Quantifies self-similarity across spatial scales
  - Computed using 10 logarithmically-spaced box sizes
- **Lacunarity**: (Variance / Mean²) + 1 of occupied boxes
  - Measures spatial heterogeneity and gaps in the vascular pattern

### Geometric Features (Tortuosity)

All tortuosity metrics are computed using arc-length parameterized cubic B-splines with weighted curvature accounting for path multiplicity:

- **Geodesic Length**: Total path length through the vessel segment (mm)
- **Spline Arc Length**: B-spline fitted arc length (mm)
- **Spline Chord Length**: Straight-line distance between endpoints (mm)
- **Mean Curvature**: ∫[κ(s)/n(s)]ds - weighted average curvature
- **Mean Square Curvature**: ∫[κ(s)²/n(s)²]ds - curvature variability
- **RMS Curvature**: √(Mean Square Curvature / Arc Length)
- **Arc-over-Chord Ratio**: Spline Arc Length / Chord Length (≥1, higher = more tortuous)
- **Fit RMSE**: Root mean square error of spline fit to original points (mm)

*Note: n(s) represents path multiplicity, accounting for segments traversed multiple times in the graph*

---

## 🔬 Technical Details

### Graph Construction Algorithm

1. **Mesh Generation**: Create surface mesh using vedo for geodesic distance computation
2. **Skeletonization**: Uses scikit-image `skeletonize` function with 3D morphological thinning
3. **Node Detection**: 26-connectivity neighborhood analysis to identify:
   - Endpoints (degree = 1)
   - Bifurcations (degree = 3)
   - Abnormal nodes (degree > 3)
4. **Edge Weighting**: Euclidean distance between adjacent nodes
5. **Orphan Handling**: Connect isolated branches within distance threshold using angle validation
6. **Laplacian Smoothing**: Iterative smoothing of node positions (alpha=0.8, iterations=2)
7. **Geodesic Distance**: Mesh-based geodesic path lengths for accurate tortuosity measurement
8. **Artifact Removal**: Selective pruning of triangular loops (3-node cycles) by removing the longest edge

### Tortuosity Computation

The tortuosity pipeline implements weighted curvature analysis:

```python
def compute_tortuosity_metrics(points, smoothing=0, n_samples=500, counts=None):
    """
    Compute arc-length parameterized tortuosity with multiplicity weighting.
    
    Parameters:
        points: (N, 3) array of vessel centerline points
        smoothing: B-spline smoothing factor (default: 0)
        n_samples: Number of arc-length samples (default: 500)
        counts: Multiplicity weights n(x) per point (optional)
    
    Returns:
        dict with 7 tortuosity metrics
    """
```

Key innovations:
- Arc-length reparameterization ensures uniform sampling along curves
- Path multiplicity weighting (n(s)) corrects for graph segments traversed multiple times
- Weighted curvature: κ(s)/n(s) properly accounts for overlapping vessel representations

### Fractal Dimension Algorithm

Box-counting method with logarithmic box sizes:

```python
def fractal_dimension(points, box_sizes=None):
    """
    Compute fractal dimension D = -Δlog(N(ε))/Δlog(ε)
    
    Uses 10 logarithmically-spaced box sizes from max_dim/50 to max_dim
    Fits linear regression to log-log plot
    """
```

---
