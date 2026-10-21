## 🚀 Installation

### Requirements
- Python 3.10 or higher
Python packages (`requirements.txt`):
- NumPy (>=1.21.0), SciPy (>=1.7.0), scikit-image (>=0.19.0), NetworkX (>=2.6.0)
- nibabel (>=3.2.0) - NIfTI input/output
- pandas (>=1.3.0) - result tables
- vedo (>=2023.4.0) - surface mesh for the edge check and VTP output
- plotly (>=5.0.0) - HTML QC report

Optional:
- matplotlib - registration snapshots (`--registration_snapshot`)
- FSL (brain extraction and registration) - **required only for atlas-based mode**

### Install from GitHub

```bash
git clone https://github.com/i-vesseg/CaravelMetrics.git
cd CaravelMetrics
pip install -r requirements.txt
```

### FSL Installation

For atlas-based regional analysis, FSL must be installed separately:
- **Linux/macOS**: Follow instructions at https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
- **Windows**: Use WSL (Windows Subsystem for Linux) and install FSL in the Linux environment

**Note**: FSL is not required for whole-brain analysis mode.

---