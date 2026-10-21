## 🔬 Features

CaravelMetrics reports, per connected component, per atlas region and for the whole network:

### Morphometric Features
- **Total Vessel Length**: summed centreline length (mm)
- **Volume**: voxel-counted mask volume when the mask is given; graph volume (`volume_approx_cylinder`) otherwise
- **Average Diameter**: length-weighted mean vessel diameter

### Topological Features
- **Bifurcation Count & Density**: junctions and junctions per mm
- **Number of Loops**: independent cycles (collateral circulation)
- **Abnormal Degree Nodes**: junctions with 4+ branches straight from skeletonisation (merged junctions are counted separately)
- **Connected Components**: network fragmentation

### Fractal Features
- **Fractal Dimension**: box-counting dimension of the whole network, on fixed 1–16 mm boxes
- **Lacunarity**: spatial heterogeneity, fixed 5 mm boxes

### Geometric Features
- **Tortuosity index** (arc/chord), **mean curvature**, **mean squared curvature**
- **Bifurcation angles** per junction

---

## 📊 Validation Results

*These IXI results were obtained with the ISBI 2026 version of the code; several feature definitions have changed since (see below).*

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

All lengths are in mm, measured along the centreline graph. Accuracy figures are from synthetic
phantoms with known geometry.

### Morphometric

- **Total length**: sum of graph edge lengths. Skeleton tips that stop short of the vessel end are
  extended (on by default); phantom length error −0.1 % on average.
- **Radius**: per node, the radius of the cylinder holding the node's share of the mask volume,
  `r = √(V / (π·l))` (V: mask voxels closest to the node; l: half its incident edge lengths),
  median-filtered over ±2 nodes. Phantom error −0.6 % ± 1.9; on real masks it matches the area of
  perpendicular cross-sections to 0–2 % (median). Nodes within 3 mm of a junction read −5 % to +1 %
  on phantoms. The inscribed-sphere (distance transform) value is kept as `radius_edt` (reads
  12–15 % low).
- **Volume**: voxel count × voxel volume when the mask is given; otherwise the sum of conical
  frustums along the edges (`volume_approx_cylinder`).
- **Average diameter**: length-weighted mean of 2 × radius.

### Topological

- **Number of bifurcations**: nodes with degree ≥ 3. **Bifurcation density**: per mm of length.
- **Number of loops**: edges − nodes + components.
- **Abnormal degree nodes**: degree ≥ 4 nodes produced by skeletonisation. Junctions merged by the
  pipeline (two junctions < 1 mm apart) are reported as `num_merged_junctions`.
- **Segments**: the graph is split into branch segments between junctions and endpoints; each
  vessel stretch appears once.

### Fractal

- **Fractal dimension**: box-counting slope of the centreline (resampled every 0.1 mm) over fixed
  box sizes 1–16 mm, on the whole network of the region / subject; `fractal_dimension_r2` is the
  fit quality. A fixed range makes values comparable across subjects.
- **Lacunarity**: variance / mean² + 1 of centreline length per fixed 5 mm box, over the bounding
  box of the network. It depends on how much of the vascular tree the field of view covers
  (cropping 15–30 % of a network changes it by about 7 %; restricting the boxes to the network's
  convex hull did not help), so compare it only between scans with matched coverage.

### Geometric

- **Tortuosity index**: arc length / chord of each segment; summary = median over segments ≥ 6 mm.
- **Mean curvature** (1/mm): curvature of a smoothing spline (σ = 0.25 mm) through each segment,
  excluding 2 mm at each end; NaN for segments < 6 mm. Summary: length-weighted mean.
  `curvature_length_coverage` is the share of length that has a value. Phantom error 0.001 /mm.
  Shorter segments cannot be measured at these voxel sizes: allowing 4–6 mm segments gave errors
  of +0.06 to +0.11 /mm on phantoms, the size of the signal itself. Segments of 6–9 mm squeezed
  between two junctions still read about +0.04 /mm high.
- **Bifurcation angle**: angle between the two daughter branches, measured 2–6 mm from the junction;
  **deflections**: angle between the parent's continuation and each daughter. Phantom error ≈ 2–3°.
  The parent is chosen by geometry first (the two daughters are the closest pair), then by width.
  Junctions whose two daughters rejoin within a loop of ≤ 25 mm perimeter are flagged
  `on_short_loop` and left out of the median (`num_bifurcations_short_loop` counts them): these are
  tunnels through the mask (a hole the skeleton goes around, reading as a 7–26° "bifurcation") or
  tiny loops, where the angle is not a branching angle. On the test masks 14–20 % of measured
  junctions are flagged, including all six tunnels found by eye, and the median angle rises by
  5–8°. The rule cannot tell a tunnel from a real small loop, so filter on the column if you need
  a different choice.

### Quality control

`*_qc.csv` lists the uncovered-vessel fraction, graph/mask volume ratio, fragmentation, spur
density, curvature coverage, FD fit quality and facing vessel ends, with `qc_pass` / `qc_flags`.
On the validation set the flags raised one false alarm on 72 good inputs (a short phantom tube
with a real wall-bump spur) and caught all 8 noisy ones.

A clean segmentation with a missing piece is harder. `facing_endpoint_pairs` counts pairs of wide
vessel ends (radius ≥ 1 mm) that point at each other across a gap of up to 12 mm, which a natural
vessel end never produces. It caught 8 of 12 test masks with a 3 or 6 mm slab removed, with no
false alarm on the 72 good inputs. It misses gaps that only interrupt vessels thinner than 1 mm
(thin-vessel dropouts are too common to flag).
