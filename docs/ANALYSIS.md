## 📊 Statistical Analysis

For population-level analysis, we provide a comprehensive Jupyter notebook (`CaravelMetrics_Stats_Analysis_IXI.ipynb`) that includes:

### Analysis Modules

1. **Dynamic Feature Detection**: Automatically adapts to available features
2. **Data Quality Assessment**: Missing data patterns, outlier detection
3. **Descriptive Statistics**: Distribution analysis, summary tables
4. **Age-Related Analysis**: 
   - Spearman correlations
   - Age quartile comparisons (ANOVA)
   - Effect size calculations (η²)
5. **Sex-Based Analysis**: T-tests with Cohen's d effect sizes
6. **Anthropometric Correlations**: Height, weight, BMI associations
7. **Multi-Center Analysis**: Site effect quantification
8. **Regional Analysis**: Territory-specific patterns
9. **Hemispheric Asymmetry**: Left vs right comparisons
10. **Advanced Analyses**: 
    - Random Forest feature importance
    - Interaction effects
    - Age-stratified subgroup analysis

### Running Statistical Analysis

```python
# Load your vessel metrics CSV files
import pandas as pd
import numpy as np

# Combine results from multiple subjects
results = []
for subject_id in subject_list:
    # Load region_summary.csv for aggregated metrics
    df = pd.read_csv(f'results/{subject_id}/region_summary.csv')
    df['subject_id'] = subject_id
    results.append(df)

combined_df = pd.concat(results, ignore_index=True)

# Merge with demographic data
metadata = pd.read_excel('IXI_METADATA.xls')
analysis_df = combined_df.merge(metadata, on='subject_id')

# Run analyses using the provided notebook
# See CaravelMetrics_Stats_Analysis_IXI.ipynb for complete examples
```

### FDR Correction

All statistical tests apply Benjamini-Hochberg False Discovery Rate (FDR) correction at q < 0.05.

```python
from statsmodels.stats.multitest import multipletests

# Example: correct multiple comparisons
p_values = [0.001, 0.045, 0.12, 0.003, 0.08]
reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
```

---

## 🎨 Visualization Examples

### Regional Feature Maps

```python
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Load registered atlas and metrics (atlas-based mode)
atlas = nib.load('results/sub001/sub001_registered_atlas.nii.gz').get_fdata()
metrics_df = pd.read_csv('results/sub001/region_summary.csv')

# Create a 3D volume with feature values per region
feature_map = np.zeros_like(atlas)
for _, row in metrics_df.iterrows():
    region_id = int(row['region_label'])
    feature_value = row['Largest_endpoint_root_mean_curvature']
    feature_map[atlas == region_id] = feature_value

# Visualize middle slice
plt.figure(figsize=(10, 8))
plt.imshow(feature_map[:, :, feature_map.shape[2]//2], cmap='hot')
plt.colorbar(label='Mean Curvature')
plt.title('Regional Tortuosity Map')
plt.axis('off')
plt.savefig('regional_tortuosity.png', dpi=300, bbox_inches='tight')
```

### Age Correlation Heatmap

```python
import seaborn as sns
from scipy.stats import spearmanr

# Compute correlations between age and features
features = ['total_length', 'volume', 'bifurcation_density', 
            'Largest_endpoint_root_mean_curvature']
correlations = {}
p_values = {}

for feature in features:
    r, p = spearmanr(analysis_df['age'], analysis_df[feature])
    correlations[feature] = r
    p_values[feature] = p

# Plot heatmap
plt.figure(figsize=(8, 6))
corr_df = pd.DataFrame([correlations]).T
sns.heatmap(corr_df, annot=True, cmap='RdBu_r', center=0, 
            vmin=-0.6, vmax=0.6, cbar_kws={'label': 'Spearman r'})
plt.title('Age Correlations with Vessel Features')
plt.tight_layout()
plt.savefig('age_correlations.png', dpi=300)
```
---
