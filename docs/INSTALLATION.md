## 🚀 Installation

### Requirements
- Python 3.10 or higher
- NumPy
- SciPy
- scikit-image
- scikit-learn
- NetworkX
- nibabel (for NIfTI file handling)
- pandas
- FSL (for brain extraction and registration - **required only for atlas-based mode**)
- nipype (Python interface to FSL - **required only for atlas-based mode**)

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