# Caravel-Metrics

<!-- Logo/Banner Image -->
<p align="center">
  <img src="logos/CaravelMetrics_LOGO.png" alt="CaravelMetrics Logo" width="300"/>
</p>

**[An Automated Framework for Large-Scale Graph-Based Cerebrovascular Analysis](https://arxiv.org/abs/2512.03869)** (Arxiv)

**Accepted at ISBI2026!**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Accepted@ISBI%202026-green.svg)](link-to-paper)

CaravelMetrics is an open-source Python framework for comprehensive, automated analysis of cerebrovascular networks from medical imaging data. It transforms binary vessel segmentation masks into quantitative morphometric, topological, and geometric features across anatomically defined brain territories.

---

## 🎯 Overview

Understanding cerebrovascular morphology is critical for detecting disease and distinguishing pathological changes from normal aging. CaravelMetrics enables large-scale population studies by automating vessel feature extraction through graph-based analysis.

<!-- Pipeline Diagram -->
<p align="center">
  <img src="logos/pipeline_pp.png" alt="CaravelMetrics Pipeline" width="800"/>
</p>


### Key Capabilities

- **Modular Architecture**: Three independent processing modules (Atlas Registration, Graph Extraction, Metric Computation)
- **Batch Processing**: Automated multi-subject pipeline with parallel execution
- **Automated Graph Construction**: Converts vessel segmentation masks to mathematical graph representations via skeletonization
- **Multi-Scale Feature Extraction**: Computes 15 complementary features across four categories
- **Dual Analysis Modes**: Supports both whole-brain and atlas-based regional analysis
- **Regional Analysis**: Integrates arterial atlas for territory-specific measurements across 30 brain regions
- **Population-Level Studies**: Validated on 570 subjects (ages 20-86) from the IXI dataset
- **Reproducible Pipeline**: End-to-end automation from segmentation masks to statistical analysis
- **Advanced Graph Processing**: Geodesic distance computation, orphan node handling, artifact removal

---

## � Quick Start

```bash
# 1. Clone and install
git clone https://github.com/i-vesseg/CaravelMetrics.git
cd CaravelMetrics
pip install -r requirements.txt

# 2. Configure your data paths in CaravelMetrics.py
# Edit: segmentation_folder, image_folder, output_base_folder, use_atlas

# 3. Run the pipeline
python CaravelMetrics.py

# The pipeline will automatically process all subjects in your data folders
```

For detailed configuration options and advanced usage, see the [Usage Guide](docs/USAGE.md).

---

## 🫀 Coronary Extension

On top of the unmodified CaravelMetrics baseline above (built for cerebrovascular/brain analysis), this repository adds a separate, additive `coronary_extension/` package for coronary artery segment and vessel analysis. None of the baseline modules are modified to build it: `Graph_extractor.extract_graph` is reused as-is, and only `Segment_extractor.build_segments`, `Metric_extractor.compute_tortuosity_metrics` and `Graph_extractor.SLICER_MATRIX` are imported and called.

### What it adds

- **`coronary_extension/Segment_extractor.py`** — an alternative, bifurcation-based segment definition: instead of the baseline's root→endpoint shortest paths (which can overlap), a segment here is a maximal chain of nodes between two topological nodes (degree ≠ 2), so every edge of the graph belongs to exactly one segment. `build_segments_merged` additionally merges away degenerate (<4-node) segments so short skeletonization artifacts don't fragment a genuine vessel.
- **`coronary_extension/Path_analyzer.py`** — answers "what is the path from the coronary ostium to this 3D point?", as an ordered chain of elementary segments with metrics. Lets a vessel of interest be defined by picking one endpoint (e.g. in 3D Slicer) instead of needing a full segmentation mask.
- **`coronary_extension/Vessel_aggregator.py`** — reassembles elementary segments into whole endpoint-to-endpoint vessels (from a voxel mask via `metrics_per_label`, or from a path query via `aggregate_path_metrics`) and computes vessel-level tortuosity metrics:
  - `arc_over_chord`, `spline_arc_length`/`spline_chord_length`, `fit_rmse` — from a single spline fit on the whole reconstructed vessel path.
  - `mean_curvature` / `mean_square_curvature` / `rms_curvature` — a length-weighted aggregation of each elementary segment's own spline fit, which avoids the curvature spike a single whole-path spline would otherwise show at internal bifurcation-joint nodes.
  - `total_turning`, `junction_turning`, `n_junctions`, `soam` (Sum-Of-Angles Metric, rad/mm) — a junction-aware turning measure that additionally recovers the turn angle concentrated exactly at a bifurcation node, which is invisible to the length-weighted curvature aggregation above (a junction node is only ever a segment endpoint, never one of its interior spline samples).
  - Automatic pruning of short, spurious dead-end branches (`MIN_BRANCH_LENGTH_MM`) so a mask that is anatomically a single vessel isn't mistakenly flagged as a branching tree.
- **`coronary_extension/Segment_visualizer.py`** — exports segments (with their metrics as cell arrays) to a colored `.vtp` file for inspection in 3D Slicer / ParaView.
- **`Batch_runner.py`** (repo root) — batch entry point: given a CSV of `(subject_id, vessel_name, x, y, z)` ostium/endpoint coordinates, runs the path-based aggregation for every row and writes one results CSV.
- **`run_demo.py`** (repo root) — end-to-end, single-subject demo exercising the whole additive pipeline (segment extraction, path query, mask-based aggregation, VTP export).

### Running the coronary demo

```bash
# Single-subject, end-to-end demo
python run_demo.py

# Batch mode over a CSV of ostium/endpoint coordinates
python Batch_runner.py data_test/endpoints_test.csv
```

### Minor fixes to the baseline

- `Graph_extractor.py`: documented the RAS+ → LPS+ (`SLICER_MATRIX`) coordinate-frame conversion applied to node positions, and which downstream code needs to invert it.
- `Metric_extractor.py`: replaced a bare `except:` in the atlas lookup with `except Exception`, chaining the raised error with `from e` so the original traceback is preserved.
- `CaravelMetrics.py`: added per-stage timing metrics and logging, and an option to save the skeleton mask as output.

---

## 📚 Documentation

**Full documentation and usage examples are available in [`docs/`](docs/)**

- **[Pipeline Overview](docs/PIPELINE.md)** - Processing steps and modular architecture
- **[Installation Guide](docs/INSTALLATION.md)** - Detailed setup instructions
- **[Usage Guide](docs/USAGE.md)** - Configuration and batch processing examples
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

## 🤝 Contributors

- **Bogdan Ion** - Data Science Department, EURECOM
- **Duccio Lalli** - Double Degree MSc Student in Data Science and Engineering at Politecnico di Torino & EURECOM

---

### Contact

For questions about the code or research collaborations:
- Open an issue on GitHub
- Email: daniele.falcetta@eurecom.fr, maria.zuluaga@eurecom.fr

---
