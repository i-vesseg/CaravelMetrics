# Caravel-Metrics

<!-- Logo/Banner Image -->
<p align="center">
  <img src="logos/CaravelMetrics_LOGO.png" alt="CaravelMetrics Logo" width="300"/>
</p>

**An Automated Framework for Large-Scale Graph-Based Cerebrovascular Analysis**

**(Submitted at ISBI2026)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Submitted@ISBI%202026-green.svg)](link-to-paper)

CaravelMetrics is an open-source Python framework for comprehensive, automated analysis of cerebrovascular networks from medical imaging data. It transforms binary vessel segmentation masks into quantitative morphometric, topological, and geometric features across anatomically defined brain territories.

---

## 🎯 Overview

Understanding cerebrovascular morphology is critical for detecting disease and distinguishing pathological changes from normal aging. CaravelMetrics enables large-scale population studies by automating vessel feature extraction through graph-based analysis.

<!-- Pipeline Diagram -->
<p align="center">
  <img src="logos/pipeline_pp.png" alt="CaravelMetrics Pipeline" width="800"/>
</p>


### Key Capabilities

- **Automated Graph Construction**: Converts vessel segmentation masks to mathematical graph representations via skeletonization
- **Multi-Scale Feature Extraction**: Computes 15 complementary features across four categories
- **Dual Analysis Modes**: Supports both whole-brain and atlas-based regional analysis
- **Regional Analysis**: Integrates arterial atlas for territory-specific measurements across 30 brain regions
- **Population-Level Studies**: Validated on 570 subjects (ages 20-86) from the IXI dataset
- **Reproducible Pipeline**: End-to-end automation from segmentation masks to statistical analysis

---

## 📚 Documentation

**Full documentation and usage examples are available in [`docs/`](docs/)**

- **[Pipeline Overview](docs/PIPELINE.md)** - Processing Steps and Modalities 
- **[Installation Guide](docs/INSTALLATION.md)** - Detailed setup instructions
- **[Usage Guide](docs/USAGE.md)** - Complete command reference and examples
- **[Features Documentation](docs/FEATURES.md)** - Detailed feature definitions
- **[Analysis Guide](docs/ANALYSIS.md)** - Statistical analysis and visualization

---

## 📄 Citation

Details will be provided after acceptance.

---
## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/license/mit) file for details.

---

## 🙏 Acknowledgments

This work is co-funded by:
- **European Research Council** (ERC CARAVEL 101171357)
- **ANR JCJC I-VESSEG** (22-CE45-0015-01)
- **TwinsUK Imaging**: A Resource for Ageing Research (Chronic Disease Research Foundation)
- **King's Health Partners Digital Health Hub** and **EPSRC**

### Data and Resources

- **IXI Dataset**: https://brain-development.org/ixi-dataset/
- **Arterial Atlas**: Liu et al., Scientific Data (2023) - https://doi.org/10.1038/s41597-023-01967-w
- **VesselVerse**: [Falcetta et al., MICCAI (2025)](https://papers.miccai.org/miccai-2025/paper/0087_paper.pdf) - Vessel segmentation resource. [(Official Website)](https://i-vesseg.github.io/vesselverse/)

---

## 👥 Authors

- **Daniele Falcetta** - Data Science Department, EURECOM
- **Liane S. Canas** - School of Biomedical Engineering & Imaging Sciences, King's College London
- **Lorenzo Suppa** - Data Science Department, EURECOM & Politecnico di Torino
- **Matteo Pentassuglia** - Data Science Department, EURECOM
- **Jon Cleary** - School of Biomedical Engineering & Imaging Sciences, King's College London
- **Marc Modat** - School of Biomedical Engineering & Imaging Sciences, King's College London
- **Sébastien Ourselin** - School of Biomedical Engineering & Imaging Sciences, King's College London
- **Maria A. Zuluaga** - Data Science Department, EURECOM & School of Biomedical Engineering, King's College London

### Contact

For questions about the code or research collaborations:
- Open an issue on GitHub
- Email: daniele.falcetta@eurecom.fr, maria.zuluaga@eurecom.fr

---
