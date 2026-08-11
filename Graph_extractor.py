import os
import argparse
import pickle
import vedo
import nibabel as nib
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, label, center_of_mass, map_coordinates, binary_fill_holes, zoom
from scipy.spatial import cKDTree
from nibabel.affines import apply_affine

# =========================================================================
#                             HELPER FUNCTIONS
# =========================================================================

def preprocess_mask(data):
    """Applies a 3D binary fill holes operation to the vascular mask."""
    # binary_fill_holes requires a boolean array and returns a boolean array.
    # We cast it back to the original data type (e.g., float) for downstream compatibility.
    return binary_fill_holes(data > 0).astype(data.dtype)


def crop_to_mask_bbox(data, affine, voxel_sizes, margin_mm=5.0):
    """
    Crop to a tight bbox around the mask (+ margin_mm) so the mostly-empty
    full FOV isn't carried through resampling/skeletonization/meshing.
    margin_mm keeps binary_fill_holes, the isosurface, and every downstream
    physical-space threshold unaffected -- only the array shrinks.
    Returns the cropped data and an affine with world coordinates unchanged.
    """
    idx = np.argwhere(data > 0)
    if idx.size == 0:
        return data, affine

    margin_vox = np.ceil(margin_mm / np.asarray(voxel_sizes)).astype(int)
    lo = np.maximum(idx.min(axis=0) - margin_vox, 0)
    hi = np.minimum(idx.max(axis=0) + margin_vox + 1, np.array(data.shape))

    cropped = data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    new_affine = affine.copy()
    new_affine[:3, 3] = affine[:3, :3] @ lo + affine[:3, 3]
    return cropped, new_affine


def resample_to_isotropic(data, affine, voxel_sizes, target_spacing=0.5):
    """
    Resample a binary mask to isotropic voxel spacing.

    Interpolates the signed distance transform (a smooth field) rather than
    the binary mask directly, then re-thresholds at 0 -- avoids the
    staircase/aliasing artifacts that thresholding a directly-interpolated
    step function produces, especially when downsampling.

    Parameters
    ----------
    data         : np.ndarray  - binary mask (any numeric dtype)
    affine       : np.ndarray  - 4x4 NIfTI affine
    voxel_sizes  : np.ndarray  - (3,) current voxel sizes in mm
    target_spacing : float     - desired isotropic voxel size in mm

    Returns
    -------
    resampled_data  : np.ndarray  - binary mask at target_spacing
    new_affine      : np.ndarray  - 4x4 updated affine
    new_voxel_sizes : np.ndarray  - (3,) all equal to target_spacing
    """
    zoom_factors = np.array(voxel_sizes) / target_spacing

    # Signed distance transform (mm), positive inside / negative outside.
    inside = distance_transform_edt(data > 0, sampling=voxel_sizes).astype(np.float32)
    outside = distance_transform_edt(data == 0, sampling=voxel_sizes).astype(np.float32)
    sdf = inside - outside
    del inside, outside

    sdf_resampled = zoom(sdf, zoom_factors, order=1)
    del sdf
    # float32, not bool/uint8: map_coordinates() later needs fractional
    # values between 0 and 1 along its sample paths.
    resampled = (sdf_resampled > 0).astype(np.float32)
    del sdf_resampled

    # Rebuild affine: preserve origin (col 3) and orientation, scale columns 0-2
    new_affine = affine.copy().astype(np.float64)
    for i in range(3):
        col_norm = np.linalg.norm(affine[:3, i])
        if col_norm > 0:
            new_affine[:3, i] = (affine[:3, i] / col_norm) * target_spacing

    new_voxel_sizes = np.array([target_spacing, target_spacing, target_spacing])
    return resampled, new_affine, new_voxel_sizes


def laplacian_smooth_graph(graph, iterations, alpha):
    """
    Smooths node positions using uniform ("umbrella") Laplacian smoothing.
    Equal-weight neighbor averaging is only valid while edge lengths stay
    comparable, which holds here since edges come from 26-connectivity on
    the voxel grid; a much-longer edge would drag its node off-centerline.
    """
    pos = {n: graph.nodes[n]['pos'] for n in graph.nodes()}
    for _ in range(iterations):
        new_pos = pos.copy()
        for node in graph.nodes():
            if graph.degree(node) <= 1: continue
            neighbors = list(graph.neighbors(node))
            if not neighbors: continue
            neighbor_sum = np.sum([pos[n] for n in neighbors], axis=0)
            new_pos[node] = (1 - alpha) * pos[node] + alpha * (neighbor_sum / len(neighbors))
        pos = new_pos
    for n, p in pos.items(): graph.nodes[n]['pos'] = p
    return graph


def extract_graph(
    nifti_path: str,
    output_folder: str,
    smooth_iters: int = 2,
    smooth_alpha: float = 0.8,
    smooth_reference_spacing_mm: float = 0.5,
    smooth_iters_max: int = 3,
    intensity_threshold: float = 0.6,
    orphan_distance_threshold: float = 2.5,
    merge_distance: float = 3.0,
    target_iso_spacing: float = None
):
    case_name = os.path.basename(output_folder)
    print(f"\n{'=' * 60}")
    print(f"PROCESSING CASE: {case_name}")
    print(f"File: {nifti_path}")
    print(f"{'=' * 60}")

    # Extract clean base name from the input file
    base_name = os.path.basename(nifti_path).split('.')[0]

    os.makedirs(output_folder, exist_ok=True)
    output_pkl = os.path.join(output_folder, f"{base_name}_vessel_data.pkl")
    output_vtp = os.path.join(output_folder, f"{base_name}_vessel_graph.vtp")

    if os.path.exists(output_pkl):
        print(f"  -> Output PKL already exists. Skipping case.")
        return output_pkl

    # -------------------------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------------------------
    print("  Loading NIFTI...")
    nii = nib.load(nifti_path)
    raw = nii.get_fdata(dtype=np.float32)
    data = (raw > 0).astype(np.uint8)  # binary mask: 1 byte/voxel instead of 4
    del raw
    affine = nii.affine.copy().astype(np.float64)
    voxel_sizes = nib.affines.voxel_sizes(affine)

    # -------------------------------------------------------------------------
    # 1.5  PREPROCESSING  –  fill internal holes
    # -------------------------------------------------------------------------
    print("  Preprocessing: Filling internal holes in the mask...")
    data = preprocess_mask(data)

    # 1.55  CROP TO MASK BOUNDING BOX -- vascular masks are often <1% of the
    # full FOV; cropping before resampling/skeletonization/meshing (all of
    # which scale with array size) is the single biggest lever on peak memory.
    orig_shape = data.shape
    data, affine = crop_to_mask_bbox(data, affine, voxel_sizes)
    print(f"  Cropped to mask bounding box: {orig_shape} -> {data.shape} "
          f"({100 * np.prod(data.shape) / np.prod(orig_shape):.1f}% of original volume)")

    # 1.6  RESAMPLE TO ISOTROPIC SPACE -- skeletonize() is sensitive to voxel
    # sampling density, so anisotropic data biases metrics along the
    # finer-resolution axes. Default target is this image's own finest native
    # axis, so every axis is only ever upsampled, never downsampled.
    if target_iso_spacing is None:
        target_iso_spacing = float(np.min(voxel_sizes))
        print(f"  Auto isotropic target: {target_iso_spacing:.3f} mm "
              f"(this image's own finest native axis).")

    if not np.allclose(voxel_sizes, target_iso_spacing, atol=0.05):
        print(f"  Resampling: {np.round(voxel_sizes, 3)} mm  →  "
              f"{target_iso_spacing:.2f} mm isotropic ...")
        data, affine, voxel_sizes = resample_to_isotropic(
            data, affine, voxel_sizes, target_spacing=target_iso_spacing
        )
        print(f"  Resampled volume shape: {data.shape}")
    else:
        print(f"  Voxel spacing {np.round(voxel_sizes, 3)} mm already close to "
              f"target {target_iso_spacing:.2f} mm — skipping resample.")

    # Resolution-scaled smoothing dose: a fixed iteration count covers less
    # physical distance at finer voxel spacing, so it under-smooths finer
    # scans relative to coarser ones. Scale iterations by
    # (reference/actual spacing)^2 to keep the physical smoothing radius
    # comparable across scans, capped at smooth_iters_max to avoid
    # over-smoothing at extreme spacing ratios.
    smooth_iters_effective = min(smooth_iters_max, max(1, round(
        smooth_iters * (smooth_reference_spacing_mm / target_iso_spacing) ** 2
    )))
    if smooth_iters_effective != smooth_iters:
        print(f"  Resolution-scaled smoothing: {smooth_iters} iterations at "
              f"{smooth_reference_spacing_mm}mm reference  →  {smooth_iters_effective} "
              f"iterations at this scan's {target_iso_spacing:.2f}mm spacing "
              f"(capped at {smooth_iters_max}).")

    # inv_affine must be computed *after* potential resampling
    inv_affine = np.linalg.inv(affine)

    # -------------------------------------------------------------------------
    # 2. MESH GENERATION
    # -------------------------------------------------------------------------
    print("  Generating Reference Mesh...")
    vol = vedo.Volume(data)
    mesh = vol.isosurface(value=0.5).apply_transform(affine.tolist())
    mesh.clean()
    del vol  # vedo/VTK holds its own copy of `data`; only `mesh` is needed from here on

    # -------------------------------------------------------------------------
    # 3. SKELETONIZATION
    # -------------------------------------------------------------------------
    print("  Skeletonizing...")
    skeleton_mask = skeletonize(data > 0)
    skel_indices = np.argwhere(skeleton_mask)
    skel_points_world = apply_affine(affine, skel_indices)

    # Distance transform for accurate radius sampling (in mm)
    full_dist_map_vox = distance_transform_edt(data > 0, sampling=voxel_sizes).astype(np.float32)

    # 4. ORPHAN DETECTION (QC diagnostic only -- not added to the graph).
    # Orphans are mask regions the skeleton never reached; stitching them
    # into the graph would fabricate off-centerline segments and false
    # bifurcations, so they're only counted as a segmentation-gap signal.
    print(f"  Detecting orphan regions (Threshold > {orphan_distance_threshold} mm)...")
    dist_to_skel = distance_transform_edt(~skeleton_mask, sampling=voxel_sizes).astype(np.float32)
    del skeleton_mask  # only needed to build `~skeleton_mask` above

    orphan_mask = (data > 0) & (dist_to_skel > orphan_distance_threshold)
    del dist_to_skel
    labeled_orphans, num_zones = label(orphan_mask)

    orphan_points = []
    for z in range(1, num_zones + 1):
        cm = center_of_mass(orphan_mask, labeled_orphans, z)
        orphan_points.append(apply_affine(affine, np.array(cm)))
    orphan_points_world = np.array(orphan_points)
    del orphan_mask, labeled_orphans

    if len(orphan_points_world) > 0:
        tree = cKDTree(orphan_points_world)
        pairs = tree.query_pairs(r=merge_distance)
        g_tmp = nx.Graph()
        g_tmp.add_nodes_from(range(len(orphan_points_world)))
        g_tmp.add_edges_from(pairs)

        merged_indices = []
        for component in nx.connected_components(g_tmp):
            comp_list = list(component)
            cluster_coords = orphan_points_world[comp_list]
            centroid = np.mean(cluster_coords, axis=0)
            closest_idx = np.argmin(np.linalg.norm(cluster_coords - centroid, axis=1))
            merged_indices.append(comp_list[closest_idx])

        if merged_indices:
            orphan_points_world = orphan_points_world[merged_indices]
        else:
            orphan_points_world = np.array([])
        print(f"    Orphan regions detected (excluded from graph): "
              f"{len(orphan_points_world)}")

    # -------------------------------------------------------------------------
    # 5. BUILD GRAPH  (with accurate radii from the distance transform)
    # -------------------------------------------------------------------------
    print("  Building Graph and Extracting Native Radii...")
    G = nx.Graph()

    # np.round, not .astype(int): the affine round-trip can land a hair below
    # the true integer index (e.g. 541.99999996), and truncating would read
    # the wrong voxel and produce a spurious zero radius.
    for i, pt in enumerate(skel_points_world):
        pos_vox = np.round(apply_affine(inv_affine, pt)).astype(int)
        pos_vox = np.clip(pos_vox, 0, np.array(data.shape) - 1)
        r = full_dist_map_vox[tuple(pos_vox)]
        G.add_node(i, pos=pt, radius=r)

    del full_dist_map_vox  # not referenced again after node radii are assigned

    idx_map = {tuple(p): i for i, p in enumerate(skel_indices)}
    offsets_26 = [
        np.array([z, y, x])
        for z in (-1, 0, 1)
        for y in (-1, 0, 1)
        for x in (-1, 0, 1)
        if not (z == 0 and y == 0 and x == 0)
    ]
    for i, curr in enumerate(skel_indices):
        for off in offsets_26:
            neigh = tuple(curr + off)
            if neigh in idx_map and idx_map[neigh] > i:
                G.add_edge(i, idx_map[neigh])

    # -------------------------------------------------------------------------
    # 6. PRUNING  –  triangle collapse
    # -------------------------------------------------------------------------
    print("  Pruning...")
    for tri in [c for c in nx.enumerate_all_cliques(G) if len(c) == 3]:
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        longest = max(
            edges,
            key=lambda e: np.linalg.norm(
                G.nodes[e[0]]['pos'] - G.nodes[e[1]]['pos']
            ) if G.has_edge(*e) else -1
        )
        if G.has_edge(*longest):
            G.remove_edge(*longest)

    # Intensity-gated edge pruning with geodesic fallback
    step = np.min(voxel_sizes) * 0.5
    edges_to_remove = []
    for u, v in G.edges():
        p1, p2 = G.nodes[u]['pos'], G.nodes[v]['pos']
        dist = np.linalg.norm(p2 - p1)
        pts = p1 + np.outer(np.linspace(0, 1, max(2, int(dist / step) + 1)), p2 - p1)
        vals = map_coordinates(data, apply_affine(inv_affine, pts).T, order=1)
        if np.any(vals < intensity_threshold):
            try:
                id1 = mesh.closest_point(p1, return_point_id=True)
                id2 = mesh.closest_point(p2, return_point_id=True)
                path_obj = mesh.geodesic(id1, id2)
                if path_obj:
                    geo_dist = np.sum(np.linalg.norm(np.diff(path_obj.vertices, axis=0), axis=1))
                    if (geo_dist / dist) > 2.0:
                        edges_to_remove.append((u, v))
                else:
                    edges_to_remove.append((u, v))
            except:
                edges_to_remove.append((u, v))
    G.remove_edges_from(edges_to_remove)
    del data  # not referenced again; remaining steps operate on the graph only

    # -------------------------------------------------------------------------
    # 6.5  FILTER SMALL COMPONENTS  (physically agnostic – works in mm)
    # -------------------------------------------------------------------------
    min_component_length_mm = 3.5
    print(f"  Filtering Components < {min_component_length_mm} mm...")

    nodes_to_remove = []
    for component in nx.connected_components(G):
        sub_G = G.subgraph(component)
        comp_length = sum(
            np.linalg.norm(G.nodes[u]['pos'] - G.nodes[v]['pos'])
            for u, v in sub_G.edges()
        )
        if comp_length < min_component_length_mm:
            nodes_to_remove.extend(list(component))

    G.remove_nodes_from(nodes_to_remove)

    # -------------------------------------------------------------------------
    # 6.6  PRUNE FALSE BRANCHES  (terminal hairs, physically agnostic)
    # -------------------------------------------------------------------------
    min_branch_length_mm = 1.5
    print(f"  Pruning terminal branches < {min_branch_length_mm} mm...")

    # Walks the whole chain from each endpoint to its nearest bifurcation
    # and prunes it if the cumulative length is under threshold -- a
    # per-edge check would miss multi-hop hairs. Uses node positions
    # directly since edges have no 'weight' attribute yet (set in section 8).
    def _euclid_weight(u, v, _d):
        return np.linalg.norm(G.nodes[u]['pos'] - G.nodes[v]['pos'])

    pruning = True
    while pruning:
        pruning = False
        endpoints = [n for n, d in G.degree() if d == 1]
        for ep in endpoints:
            if ep not in G:
                continue
            try:
                dist, path = nx.single_source_dijkstra(
                    G, ep, cutoff=min_branch_length_mm, weight=_euclid_weight
                )
            except nx.NodeNotFound:
                continue
            bif_targets = [n for n in dist if n != ep and G.degree(n) >= 3]
            if not bif_targets:
                continue
            nearest_bif = min(bif_targets, key=lambda n: dist[n])
            chain = path[nearest_bif][:-1]  # exclude the bifurcation itself
            G.remove_nodes_from(chain)
            pruning = True

    # 6.7  CONTRACT LADDER RUNGS -- skeletonize() can produce a "ladder": two
    # parallel skeleton lines cross-linked by tiny rungs. A rung's signature
    # (unlike a real small anastomosis) is both endpoints already being
    # branch points, nearly touching -- safe to auto-collapse.
    max_rung_length_mm = 1.0
    print(f"  Contracting ladder rungs < {max_rung_length_mm} mm (branch-point-to-branch-point)...")

    contracting = True
    while contracting:
        contracting = False
        for u, v in list(G.edges()):
            if u not in G or v not in G:
                continue
            if G.degree(u) >= 3 and G.degree(v) >= 3 and _euclid_weight(u, v, None) < max_rung_length_mm:
                new_pos = (G.nodes[u]['pos'] + G.nodes[v]['pos']) / 2.0
                new_radius = (G.nodes[u].get('radius', 0.5) + G.nodes[v].get('radius', 0.5)) / 2.0
                nx.contracted_nodes(G, u, v, self_loops=False, copy=False)
                G.nodes[u]['pos'] = new_pos
                G.nodes[u]['radius'] = new_radius
                contracting = True
                break  # topology changed; restart the scan

    # -------------------------------------------------------------------------
    # 7. SMOOTHING
    # -------------------------------------------------------------------------
    print(f"  Smoothing ({smooth_iters_effective} iterations)...")
    G = laplacian_smooth_graph(G, iterations=smooth_iters_effective, alpha=smooth_alpha)

    # -------------------------------------------------------------------------
    # 8. SAVE OUTPUTS
    # -------------------------------------------------------------------------
    print("  Saving Results...")
    for u, v in G.edges():
        G[u][v]['weight'] = np.linalg.norm(G.nodes[u]['pos'] - G.nodes[v]['pos'])

    # Temporary legacy map to prevent breaking Metric_extractor until rewritten
    legacy_radius_map = {n: G.nodes[n]['radius'] for n in G.nodes()}

    package = {
        'graph': G,
        'node_radius_map': legacy_radius_map,
        'voxel_sizes': voxel_sizes,          # isotropic spacing stored here
        'input_file': nifti_path
    }
    with open(output_pkl, 'wb') as f:
        pickle.dump(package, f)

    # --- EXPORT VTP FILES ---
    s_pts = [G.nodes[u]['pos'] for u, v in G.edges()]
    e_pts = [G.nodes[v]['pos'] for u, v in G.edges()]

    if s_pts:
        # 1. Native VTP (ParaView / ITK-SNAP / standard VTK)
        lines_native = vedo.Lines(s_pts, e_pts).c('green').lw(3)
        lines_native.write(output_vtp)
        print(f"    -> Saved PKL:        {output_pkl}")
        print(f"    -> Saved Native VTP: {output_vtp}")

        # 2. Slicer-specific VTP (applies LPS-to-RAS flip)
        slicer_flip = np.array([-1.0, -1.0, 1.0])
        s_pts_slicer = [p * slicer_flip for p in s_pts]
        e_pts_slicer = [p * slicer_flip for p in e_pts]

        output_vtp_slicer = output_vtp.replace(".vtp", "_slicer.vtp")
        lines_slicer = vedo.Lines(s_pts_slicer, e_pts_slicer).c('red').lw(3)
        lines_slicer.write(output_vtp_slicer)
        print(f"    -> Saved Slicer VTP: {output_vtp_slicer}")
    else:
        print("    ! Warning: Graph is empty, no VTP saved.")

    return output_pkl


# =============================================================================
#                             CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vessel Graph Extractor Module")

    parser.add_argument("--mask",        required=True,
                        help="Path to the binary vessel mask NIfTI file.")
    parser.add_argument("--output_dir",  required=True,
                        help="Directory to save the resulting .pkl and .vtp files.")

    parser.add_argument("--smooth_iters",   type=int,   default=2,
                        help="Laplacian smoothing iterations at the reference resolution "
                             "(default: 2); scaled per-scan to keep physical smoothing "
                             "radius comparable across resolutions.")
    parser.add_argument("--smooth_alpha",   type=float, default=0.8,
                        help="Laplacian smoothing alpha (default: 0.8).")
    parser.add_argument("--smooth_reference_spacing", type=float, default=0.5,
                        help="Reference isotropic spacing (mm) that --smooth_iters is "
                             "calibrated for (default: 0.5).")
    parser.add_argument("--intensity_thresh", type=float, default=0.6,
                        help="Intensity threshold for edge pruning (default: 0.6).")
    parser.add_argument("--orphan_dist",    type=float, default=2.5,
                        help="Orphan detection distance threshold mm (default: 2.5).")
    parser.add_argument("--merge_dist",     type=float, default=3.0,
                        help="Orphan merge distance threshold mm (default: 3.0).")
    parser.add_argument("--target_iso",     type=float, default=None,
                        help="Target isotropic voxel spacing in mm. Default: auto -- "
                             "uses this image's own finest native axis, so no axis is "
                             "ever downsampled. Pass an explicit value to force a fixed "
                             "common spacing across all images instead.")

    args = parser.parse_args()

    try:
        extract_graph(
            nifti_path=args.mask,
            output_folder=args.output_dir,
            smooth_iters=args.smooth_iters,
            smooth_alpha=args.smooth_alpha,
            smooth_reference_spacing_mm=args.smooth_reference_spacing,
            intensity_threshold=args.intensity_thresh,
            orphan_distance_threshold=args.orphan_dist,
            merge_distance=args.merge_dist,
            target_iso_spacing=args.target_iso
        )
    except Exception as e:
        print(f"Error during graph extraction: {e}")
        raise