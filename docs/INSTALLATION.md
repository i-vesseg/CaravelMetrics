## 🚀 Installation

### Requirements
- Python 3.10 or higher
- NumPy (>=1.21.0)
- SciPy (>=1.7.0)
- scikit-image (>=0.19.0)
- scikit-learn (>=1.0.0)
- NetworkX (>=2.6.0)
- nibabel (>=3.2.0) - for NIfTI file handling
- pandas (>=1.3.0)
- vedo - for mesh generation and geodesic distance computation
- SimpleITK - for advanced image processing
- tqdm - for progress bars
- FSL (for brain extraction and registration - **required only for atlas-based mode**)
- nipype (>=1.8.0) - Python interface to FSL (**required only for atlas-based mode**)

### Install from GitHub

```bash
git clone https://github.com/i-vesseg/CaravelMetrics.git
cd CaravelMetrics
pip install -r requirements.txt
```

### Additional Dependencies

For visualization and mesh processing:
```bash
pip install vedo SimpleITK
```

### FSL Installation

For atlas-based regional analysis, FSL must be installed separately:
- **Linux/macOS**: Follow instructions at https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
- **Windows**: Use WSL (Windows Subsystem for Linux) and install FSL in the Linux environment

**Note**: FSL is not required for whole-brain analysis mode.

---