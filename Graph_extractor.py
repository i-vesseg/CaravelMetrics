import os
import argparse
import pickle
import vedo
import nibabel as nib
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, map_coordinates, binary_fill_holes, zoom
from scipy.spatial import cKDTree
from nibabel.affines import apply_affine

# =========================================================================
#                             HELPER FUNCTIONS
# =========================================================================

def prune_terminal_hairs(G, min_branch_length_mm):
    """Remove leaf-to-bifurcation "hairs" shorter than min_branch_length_mm,
    in place. Walks each degree-1 endpoint's chain to its nearest bifurcation
    and prunes it if the cumulative length is under threshold (a per-edge
    check would miss multi-hop hairs).

    Terminal-only by construction: candidates are found by Dijkstra from a
    degree-1 leaf, so a bifurcation-to-bifurcation connector is never a
    candidate, however short. A flat mm threshold (not radius-scaled) is
    deliberate: skeleton spurs are a roughly fixed physical size, so scaling
    with caliber would delete genuine small distal vessels near thick trunks.

    Uses node positions directly (not the 'weight' edge attribute, which is
    only set later in the pipeline).
    """
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
    return G


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
    Resample a binary mask to isotropic voxel spacing, returning
    (resampled_mask, new_affine, new_voxel_sizes).

    Interpolates the signed distance transform (a smooth field) and
    re-thresholds at 0, rather than interpolating the binary step directly --
    this avoids the staircase/aliasing artifacts of the latter.
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


# Minimum smoothing sigma in voxels. Boundary-voxel noise is a voxel-scale
# feature: below ~0.7 voxel the Gaussian leaves single flipped voxels in place
# (0.25 mm on 0.6-0.7 mm voxels changed nothing, even on noisy masks), while
# 0.7 voxel fixed noisy phantom trees on 0.42, 0.6 and 0.7 mm grids.
MIN_MASK_SMOOTHING_VOX = 0.7


def effective_mask_smoothing(sigma_mm, voxel_sizes):
    """Sigma actually applied: the requested one, but at least
    MIN_MASK_SMOOTHING_VOX voxels (0 stays 0 = off)."""
    if not sigma_mm or sigma_mm <= 0:
        return 0.0
    return float(max(sigma_mm, MIN_MASK_SMOOTHING_VOX * float(np.min(voxel_sizes))))


def smooth_mask(data, voxel_sizes, sigma_mm):
    """Gaussian-smooth the signed distance (mm) of a binary mask and
    re-threshold at 0. extract_graph passes effective_mask_smoothing().

    For NOISY segmentations only (off by default). Validated on
    phantoms and test masks: with sigma = 0.25 mm, a phantom
    tree with 10-20 % of its boundary voxels flipped goes from 25-50
    bifurcations back to the true 7, and a noisy coronary mask from +23 %
    length to +1 %. The cost: vessels thinner than about one voxel are eroded
    (a 0.35 mm-radius tube keeps 30 % of its length, 0.45 mm keeps 82 %), and a
    clean brain TOF mask loses 3 % of its length. sigma = 0.5 mm erases small
    vessels entirely. Clean (e.g. nnU-Net) masks do not need it."""
    from scipy.ndimage import gaussian_filter
    vs = np.asarray(voxel_sizes, float)
    m = data > 0.5
    sdf = (distance_transform_edt(m, sampling=vs) - distance_transform_edt(~m, sampling=vs)).astype(np.float32)
    sdf = gaussian_filter(sdf, sigma=sigma_mm / vs)
    return (sdf > 0).astype(np.float32)


def equivalent_cylinder_radii(G, mask_points_world, voxel_volume, edt_radius,
                              max_factor=1.6, window=5):
    """Per-node radius r_i = sqrt(V_i / (pi * l_i)) -- the radius of the
    cylinder that has the node's share of the mask volume.

    V_i: volume of the mask voxels whose nearest graph node is i, counting
         only voxels within max_factor * EDT radius + 1 voxel of the node so
         that eroded skeleton tips do not absorb the surrounding mask.
    l_i: half the summed length of the node's incident edges.

    The EDT value at the skeleton voxel underestimates the radius by ~10%
    (the voxel sits off the true axis, and EDT measures to background voxel
    centres); this estimator is unbiased on phantoms.
    Per-node Voronoi cells are thin slices, so the raw value is
    median-filtered over the degree-2 nodes within window//2 hops. Falls back
    to the EDT radius where no volume or length is available. A tuple of
    windows returns one map per window.
    """
    nodes = list(G.nodes())
    windows = window if isinstance(window, (tuple, list)) else (window,)
    if not nodes or len(mask_points_world) == 0:
        fallback = [{n: float(edt_radius[n]) for n in nodes} for _ in windows]
        return fallback if isinstance(window, (tuple, list)) else fallback[0]
    index = {n: i for i, n in enumerate(nodes)}
    P = np.array([G.nodes[n]['pos'] for n in nodes])
    r_edt = np.array([float(edt_radius[n]) for n in nodes])
    d, j = cKDTree(P).query(mask_points_world)
    vox = float(np.cbrt(voxel_volume))
    ok = d <= max_factor * r_edt[j] + vox
    V = np.bincount(j[ok], minlength=len(nodes)) * voxel_volume
    l = np.zeros(len(nodes))
    for u, v in G.edges():
        e = np.linalg.norm(G.nodes[u]['pos'] - G.nodes[v]['pos'])
        l[index[u]] += e / 2.0
        l[index[v]] += e / 2.0
    raw = np.where((l > 0) & (V > 0), np.sqrt(V / (np.pi * np.maximum(l, 1e-12))), r_edt)
    outs = []
    for w in windows:
        half = w // 2
        out = {}
        for n in nodes:
            near = nx.single_source_shortest_path_length(G, n, cutoff=half)
            vals = [raw[index[m]] for m in near if G.degree(m) == 2] or [raw[index[n]]]
            out[n] = float(np.median(vals))
        outs.append(out)
    return outs if isinstance(window, (tuple, list)) else outs[0]


class SurfaceGeodesic:
    """Shortest path length along the edges of a surface mesh, between the
    mesh vertices nearest to two points -- the same quantity as vedo's
    Mesh.geodesic (VTK Dijkstra on edge lengths, endpoints snapped to the
    nearest vertex), but answering the only question the pipeline asks:
    "is it longer than `limit`?". Dijkstra stops at `limit`, so each query
    only explores the neighbourhood of the edge instead of the whole mesh
    (~0.1 s per query with vedo on a brain mesh, ~1 ms here).

    Vertices on two separate surface pieces return 0 (= "connected"), as the
    vedo-based check did. Separate pieces next to adjacent skeleton voxels are
    almost always diagonal-only voxel contacts inside one thin vessel:
    marching cubes splits them, but the mask (and the skeleton graph) is
    26-connected. Treating them as "no path" would cut the network into its
    6-connected pieces (ADAM test mask: 7 -> 10 components, brain: -3 %
    length).
    """

    def __init__(self, mesh):
        from scipy.sparse import coo_matrix
        V = np.asarray(mesh.vertices, dtype=np.float64)
        faces = [f for f in mesh.cells if len(f) >= 2]
        a, b = [], []
        for f in faces:
            for i in range(len(f)):
                a.append(f[i])
                b.append(f[(i + 1) % len(f)])
        a, b = np.asarray(a), np.asarray(b)
        w = np.linalg.norm(V[a] - V[b], axis=1)
        n = len(V)
        g = coo_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
        g.sum_duplicates()
        # duplicate (shared) edges were summed above; restore single lengths
        g.data = np.linalg.norm(V[g.indices] - V[np.repeat(np.arange(n), np.diff(g.indptr))], axis=1)
        self.graph = g
        self.tree = cKDTree(V)
        from scipy.sparse.csgraph import connected_components
        self.piece = connected_components(g, directed=False)[1]

    def length(self, p1, p2, limit):
        from scipy.sparse.csgraph import dijkstra
        i = int(self.tree.query(p1)[1])
        j = int(self.tree.query(p2)[1])
        if i == j or self.piece[i] != self.piece[j]:
            return 0.0
        d = dijkstra(self.graph, indices=i, limit=limit)
        return float(d[j])


def code_version():
    """Short git commit of this file's repository (+ '-dirty' if it has
    uncommitted changes), or 'unknown'. Stored with every output so results
    can be traced back to the code that produced them."""
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.run(['git', '-C', here, 'rev-parse', '--short', 'HEAD'], capture_output=True,
                             text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(['git', '-C', here, 'status', '--porcelain', '--untracked-files=no'],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return (sha + ('-dirty' if dirty else '')) if sha else 'unknown'
    except Exception:
        return 'unknown'


def extend_endpoints(G, mask_index, affine, crop_shape, min_extension_mm=0.5):
    """Extend skeleton tips that stop short of the end of their vessel.

    skimage's thinning erodes near-axis vessel ends (up to ~7 mm on a 50 mm
    phantom tube; ~1-2 % of total length on the test masks). For every
    endpoint e with EDT radius r_e, take the mask voxels whose nearest graph
    node is e, that lie more than r_e + 1 voxel from it (the part of the
    vessel the skeleton did not reach), AND that are connected to the tip
    through the mask -- otherwise the uncovered end of a *neighbouring*
    vessel can capture the tip. Extend e straight toward their centroid by
    (furthest projection - r_e), i.e. up to one radius short of the mask end
    (where the centreline of a rounded vessel end lies), and stop earlier if
    the line would leave the mask. Flat cut ends get ~0 extension.
    New nodes every voxel, radius r_e. Returns the total length added (mm).
    """
    from scipy.ndimage import label as _label
    ends = [n for n in G.nodes() if G.degree(n) == 1]
    if not ends or len(mask_index) == 0:
        return 0.0
    vs = float(np.cbrt(abs(np.linalg.det(affine[:3, :3]))))
    inv = np.linalg.inv(affine)
    W_all = apply_affine(affine, mask_index)
    inside = np.zeros(crop_shape, bool)
    inside[tuple(mask_index.T)] = True
    nodes = list(G.nodes())
    P = np.array([G.nodes[n]['pos'] for n in nodes])
    d, j = cKDTree(P).query(W_all)
    index = {n: i for i, n in enumerate(nodes)}
    next_id = max(nodes) + 1 if nodes else 0
    added = 0.0

    def _inside(p):
        k = np.round(apply_affine(inv, p)).astype(int)
        return bool(np.all(k >= 0) and np.all(k < crop_shape) and inside[tuple(k)])

    for e in ends:
        i = index[e]
        r_e = float(G.nodes[e].get('radius', 0.0))
        cell = np.flatnonzero(j == i)
        beyond = cell[d[cell] > r_e + vs]
        if len(beyond) < 3:
            continue
        # keep only the part of the cell connected (through the mask) to the
        # voxel nearest the tip
        ci = mask_index[cell]
        lo = ci.min(axis=0)
        box = np.zeros(ci.max(axis=0) - lo + 1, bool)
        box[tuple((ci - lo).T)] = True
        lab, _ = _label(box, np.ones((3, 3, 3)))
        seed = ci[np.argmin(d[cell])] - lo
        keep = lab[tuple((mask_index[beyond] - lo).T)] == lab[tuple(seed)]
        beyond = beyond[keep]
        if len(beyond) < 3:
            continue
        pe = np.asarray(G.nodes[e]['pos'], float)
        W = W_all[beyond]
        u = W.mean(axis=0) - pe
        nu = np.linalg.norm(u)
        if nu == 0:
            continue
        u /= nu
        ext = float(np.max((W - pe) @ u)) - r_e
        # stop where the straight extension would leave the mask
        t, step = 0.0, 0.25 * vs
        while t + step <= ext and _inside(pe + (t + step) * u):
            t += step
        ext = t
        if ext < min_extension_mm:
            continue
        n_new = max(1, int(np.ceil(ext / vs)))
        prev = e
        for k in range(1, n_new + 1):
            G.add_node(next_id, pos=pe + u * ext * k / n_new, radius=r_e, extended=True)
            G.add_edge(prev, next_id)
            prev = next_id
            next_id += 1
        added += ext
    return added


def facing_endpoint_pairs(G, max_gap_mm=12.0, max_angle_deg=60.0, min_radius_mm=1.0,
                          tangent_skip_mm=1.0, tangent_mm=4.0):
    """Pairs of vessel ends that point at each other across a short gap: the
    signature of a vessel interrupted by a segmentation gap. A natural vessel
    end (taper, field-of-view cut) has no second end facing it.

    Both ends must have radius >= min_radius_mm (thin-vessel dropouts are
    common and not what this counts), lie within max_gap_mm, and their
    outward directions must point along the line joining them within
    max_angle_deg. The direction is read between tangent_skip_mm and
    tangent_skip_mm + tangent_mm from the end: at a cut end the skeleton bends
    toward the centre of the flat face over the last ~1 mm. Ends of the same
    component count only if the graph path between them is > 3x the gap
    (otherwise it is a loop, not a gap). Returns (node_a, node_b, gap_mm)."""
    def _p(n):
        return np.asarray(G.nodes[n]['pos'], float)
    ends = [n for n in G if G.degree(n) == 1 and float(G.nodes[n].get('radius', 0.0)) >= min_radius_mm]
    if len(ends) < 2:
        return []
    out_dir = {}
    for e in ends:
        prev, cur, L, a = None, e, 0.0, None
        while L < tangent_skip_mm + tangent_mm:
            nb = [m for m in G.neighbors(cur) if m != prev]
            if len(nb) != 1:
                break
            L += np.linalg.norm(_p(nb[0]) - _p(cur))
            prev, cur = cur, nb[0]
            if a is None and L >= tangent_skip_mm:
                a = cur
        if a is None or a == cur:
            continue
        v = _p(a) - _p(cur)
        nv = np.linalg.norm(v)
        if nv > 0:
            out_dir[e] = v / nv
    ends = [e for e in ends if e in out_dir]
    if len(ends) < 2:
        return []
    comp = {}
    for i, c in enumerate(nx.connected_components(G)):
        for n in c:
            comp[n] = i
    cos_max = np.cos(np.radians(max_angle_deg))
    P = np.array([_p(e) for e in ends])
    pairs = []
    for i, j in sorted(cKDTree(P).query_pairs(max_gap_mm)):
        a, b = ends[i], ends[j]
        v = P[j] - P[i]
        gap = float(np.linalg.norm(v))
        if gap == 0:
            continue
        u = v / gap
        if np.dot(out_dir[a], u) < cos_max or np.dot(out_dir[b], -u) < cos_max:
            continue
        if comp[a] == comp[b]:
            try:
                if nx.shortest_path_length(G, a, b, weight=lambda x, y, _d: float(np.linalg.norm(_p(x) - _p(y)))) <= 3 * gap:
                    continue
            except nx.NetworkXNoPath:
                pass
        pairs.append((a, b, gap))
    return pairs


def graph_qc(G, mask_index, mask_points_world, voxel_volume, crop_shape, voxel_size):
    """Quality-control numbers for one extracted graph.

    uncovered_*: mask voxels farther from the graph than (EDT radius of the
        nearest node + 1 mm) -- vessel the centreline does not reach (receded
        tips, skeleton gaps). Radius-aware, unlike the previous fixed 2.5 mm
        "orphan" rule, which also flagged the outer shell of thick vessels.
    graph_to_mask_volume: frustum volume of the graph / mask volume.
    small_component_length_fraction: share of length in components < 10 mm.
    short_terminal_branches: terminal segments < 3 mm (spur proxy).
    facing_endpoint_pairs: wide vessel ends (r >= 1 mm) facing each other
        across a gap of <= 12 mm (see facing_endpoint_pairs()).
    """
    from scipy.ndimage import label as _label, find_objects
    qc = {}
    nodes = list(G.nodes())
    qc['mask_volume_mm3'] = float(len(mask_index) * voxel_volume)
    if not nodes:
        return qc
    P = np.array([G.nodes[n]['pos'] for n in nodes])
    r_edt = np.array([float(G.nodes[n].get('radius_edt', G.nodes[n].get('radius', 0.0))) for n in nodes])
    d, j = cKDTree(P).query(mask_points_world)
    unc = d > r_edt[j] + 1.0
    qc['uncovered_volume_fraction'] = float(unc.mean()) if len(unc) else 0.0
    vol = np.zeros(crop_shape, bool)
    vol[tuple(mask_index[unc].T)] = True
    lab, _ = _label(vol, np.ones((3, 3, 3)))
    ext = []
    for sl in find_objects(lab):
        if sl is not None:
            ext.append(max(s.stop - s.start for s in sl) * voxel_size)
    ext = np.array(ext)
    qc['uncovered_regions_gt2mm'] = int((ext > 2).sum())
    qc['uncovered_extent_gt2mm_mm'] = float(ext[ext > 2].sum())
    vol_g = 0.0
    for u, v in G.edges():
        l = np.linalg.norm(G.nodes[u]['pos'] - G.nodes[v]['pos'])
        ru, rv = G.nodes[u].get('radius', 0.0), G.nodes[v].get('radius', 0.0)
        vol_g += np.pi * l * (ru * ru + ru * rv + rv * rv) / 3.0
    qc['graph_to_mask_volume'] = float(vol_g / qc['mask_volume_mm3']) if qc['mask_volume_mm3'] > 0 else np.nan
    lens = []
    for c in nx.connected_components(G):
        lens.append(sum(np.linalg.norm(G.nodes[u]['pos'] - G.nodes[v]['pos']) for u, v in G.subgraph(c).edges()))
    lens = np.array(lens)
    qc['num_components'] = int(len(lens))
    qc['small_component_length_fraction'] = float(lens[lens < 10].sum() / lens.sum()) if lens.sum() > 0 else np.nan
    short = 0
    for e in [n for n in G if G.degree(n) == 1]:
        prev, cur, L = None, e, 0.0
        while True:
            nb = [m for m in G.neighbors(cur) if m != prev]
            if len(nb) != 1:
                break
            L += np.linalg.norm(G.nodes[cur]['pos'] - G.nodes[nb[0]]['pos'])
            prev, cur = cur, nb[0]
            if G.degree(cur) != 2 or L >= 3.0:
                break
        if G.degree(cur) >= 3 and L < 3.0:
            short += 1
    qc['short_terminal_branches'] = int(short)
    qc['facing_endpoint_pairs'] = len(facing_endpoint_pairs(G))
    return qc


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
    target_iso_spacing: float = None,
    min_branch_length_mm: float = 2.0,
    mask_smoothing_mm: float = 0.0,
    extend_tips: bool = True,
):
    """Extract the centreline graph of a binary vessel mask."""
    import time as _time
    _t0 = _time.time()
    params = dict(smooth_iters=smooth_iters, smooth_alpha=smooth_alpha,
                  smooth_reference_spacing_mm=smooth_reference_spacing_mm, smooth_iters_max=smooth_iters_max,
                  intensity_threshold=intensity_threshold, target_iso_spacing=target_iso_spacing,
                  min_branch_length_mm=min_branch_length_mm, mask_smoothing_mm=mask_smoothing_mm,
                  extend_tips=extend_tips)
    stages = {'rung_contractions': 0, 'tip_extension_mm': 0.0}
    case_name = os.path.basename(output_folder)
    print(f"\n{'=' * 60}")
    print(f"PROCESSING CASE: {case_name}")
    print(f"File: {nifti_path}")
    print(f"{'=' * 60}")

    base_name = os.path.basename(nifti_path).split('.')[0]

    os.makedirs(output_folder, exist_ok=True)
    output_pkl = os.path.join(output_folder, f"{base_name}_vessel_data.pkl")
    output_vtp = os.path.join(output_folder, f"{base_name}_vessel_graph.vtp")
    output_qc = os.path.join(output_folder, f"{base_name}_graph_qc.json")

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
    # 1.55  CROP TO MASK BOUNDING BOX -- vascular masks are often <1% of the
    # full FOV; cropping before resampling/skeletonization/meshing (all of
    # which scale with array size) is the single biggest lever on peak memory.
    # Done before hole filling: with an empty margin all round, every
    # background region outside the mask still touches the crop border, so
    # binary_fill_holes gives the same result on a fraction of the volume.
    orig_shape = data.shape
    data, affine = crop_to_mask_bbox(data, affine, voxel_sizes)
    print(f"  Cropped to mask bounding box: {orig_shape} -> {data.shape} "
          f"({100 * np.prod(data.shape) / np.prod(orig_shape):.1f}% of original volume)")
    print("  Preprocessing: Filling internal holes in the mask...")
    data = preprocess_mask(data)

    # 1.6  RESAMPLE TO ISOTROPIC SPACE -- skeletonize() is sensitive to voxel
    # sampling density, so anisotropic data biases metrics. Default target is
    # this image's finest native axis, so no axis is ever downsampled.
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

    mask_smoothing_used = effective_mask_smoothing(mask_smoothing_mm, voxel_sizes)
    params['mask_smoothing_mm_used'] = mask_smoothing_used
    if mask_smoothing_used > 0:
        note = "" if mask_smoothing_used == mask_smoothing_mm else \
            f" (raised from {mask_smoothing_mm} mm to {MIN_MASK_SMOOTHING_VOX} voxel)"
        print(f"  Smoothing mask boundary (sigma {mask_smoothing_used:.3f} mm{note})...")
        data = smooth_mask(data, voxel_sizes, mask_smoothing_used)

    # Resolution-scaled smoothing dose: scale iterations by
    # (reference/actual spacing)^2 so the physical smoothing radius stays
    # comparable across scans, capped to avoid over-smoothing at fine spacing.
    smooth_iters_effective = 0 if smooth_iters <= 0 else min(smooth_iters_max, max(1, round(
        smooth_iters * (smooth_reference_spacing_mm / target_iso_spacing) ** 2
    )))
    params.update(target_iso_spacing_used=float(target_iso_spacing), smooth_iters_effective=smooth_iters_effective)
    if smooth_iters_effective != smooth_iters:
        print(f"  Resolution-scaled smoothing: {smooth_iters} iterations at "
              f"{smooth_reference_spacing_mm}mm reference  →  {smooth_iters_effective} "
              f"iterations at this scan's {target_iso_spacing:.2f}mm spacing "
              f"(capped at {smooth_iters_max}).")

    # inv_affine must be computed *after* potential resampling
    inv_affine = np.linalg.inv(affine)

    # -------------------------------------------------------------------------
    # 3. SKELETONIZATION
    # -------------------------------------------------------------------------
    print("  Skeletonizing...")
    skeleton_mask = skeletonize(data > 0)
    skel_indices = np.argwhere(skeleton_mask)
    skel_points_world = apply_affine(affine, skel_indices)

    # Distance transform for accurate radius sampling (in mm)
    full_dist_map_vox = distance_transform_edt(data > 0, sampling=voxel_sizes).astype(np.float32)

    del skeleton_mask

    # -------------------------------------------------------------------------
    # 5. BUILD GRAPH  (with accurate radii from the distance transform)
    # -------------------------------------------------------------------------
    print("  Building Graph and Extracting Native Radii...")
    G = nx.Graph()

    # np.round, not .astype(int): the affine round-trip can land a hair below
    # the true integer index (e.g. 541.99999996), and truncating would read
    # the wrong voxel and produce a spurious zero radius.
    for i, pt in enumerate(skel_points_world):
        G.add_node(i, pos=pt, radius=full_dist_map_vox[tuple(skel_indices[i])])

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
    _e0 = G.number_of_edges()
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

    stages['triangle_edges_removed'] = _e0 - G.number_of_edges()
    # Intensity-gated edge pruning with geodesic fallback
    # An edge whose straight line leaves the mask is cut if the shortest path
    # along the vessel surface between its ends is more than 2x the edge
    # length. Ends on two separate surface pieces count as connected (see
    # SurfaceGeodesic): almost always a diagonal voxel contact in one vessel.
    step = np.min(voxel_sizes) * 0.5
    edges_to_remove = []
    geodesic = None
    for u, v in G.edges():
        p1, p2 = G.nodes[u]['pos'], G.nodes[v]['pos']
        dist = np.linalg.norm(p2 - p1)
        pts = p1 + np.outer(np.linspace(0, 1, max(2, int(dist / step) + 1)), p2 - p1)
        # float output: on a uint8 volume (no resampling) map_coordinates
        # would round the interpolated values to 0/1
        vals = map_coordinates(data, apply_affine(inv_affine, pts).T, order=1, output=np.float32)
        if np.any(vals < intensity_threshold):
            if geodesic is None:
                # surface mesh, built lazily: most masks need few queries
                print("  Building the surface mesh for the edge check...")
                vol = vedo.Volume(data)
                mesh = vol.isosurface(value=0.5).apply_transform(affine.tolist())
                mesh.clean()
                del vol
                geodesic = SurfaceGeodesic(mesh)
                del mesh
            try:
                if geodesic.length(p1, p2, limit=2.0 * dist) > 2.0 * dist:
                    edges_to_remove.append((u, v))
            except Exception as e:
                # Keep the edge: an unexpected error is not evidence that the
                # edge leaves the vessel.
                print(f"    ! Geodesic check failed for edge ({u}, {v}): {e}; edge kept.")
    G.remove_edges_from(edges_to_remove)
    del geodesic
    stages['surface_check_edges_cut'] = len(edges_to_remove)
    # Mask voxels (indices + world mm) for tip extension, radii and QC; the
    # volume itself is no longer needed.
    mask_index = np.argwhere(data > 0.5)
    mask_points_world = apply_affine(affine, mask_index)
    voxel_volume = float(abs(np.linalg.det(affine[:3, :3])))
    crop_shape = data.shape
    del data

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
    stages['small_component_nodes_removed'] = len(nodes_to_remove)

    # -------------------------------------------------------------------------
    # 6.6  PRUNE FALSE BRANCHES  (terminal hairs, physically agnostic)
    # -------------------------------------------------------------------------
    print(f"  Pruning terminal branches < {min_branch_length_mm} mm...")
    _n0 = G.number_of_nodes()
    prune_terminal_hairs(G, min_branch_length_mm)
    stages['hair_nodes_pruned'] = _n0 - G.number_of_nodes()

    # 6.65 EXTEND RECEDED TIPS -- see extend_endpoints(). On by default: on
    # 35 phantoms it takes the mean length error from -2.4 % to -0.1 % (worst
    # -15 % -> -2.4 %) without changing topology.
    if extend_tips:
        added = extend_endpoints(G, mask_index, affine, crop_shape)
        stages['tip_extension_mm'] = round(float(added), 3)
        print(f"  Extended receded vessel tips by {added:.1f} mm in total.")

    # 6.7  CONTRACT LADDER RUNGS -- skeletonize() can produce a "ladder": two
    # parallel skeleton lines cross-linked by tiny rungs, both endpoints
    # already branch points and nearly touching -- safe to auto-collapse.
    max_rung_length_mm = 1.0
    print(f"  Contracting ladder rungs < {max_rung_length_mm} mm (branch-point-to-branch-point)...")

    def _euclid_weight(u, v, _d=None):
        return np.linalg.norm(G.nodes[u]['pos'] - G.nodes[v]['pos'])

    contracting = True
    while contracting:
        contracting = False
        for u, v in list(G.edges()):
            if u not in G or v not in G:
                continue
            if G.degree(u) >= 3 and G.degree(v) >= 3 and _euclid_weight(u, v) < max_rung_length_mm:
                new_pos = (G.nodes[u]['pos'] + G.nodes[v]['pos']) / 2.0
                new_radius = (G.nodes[u].get('radius', 0.5) + G.nodes[v].get('radius', 0.5)) / 2.0
                nx.contracted_nodes(G, u, v, self_loops=False, copy=False)
                G.nodes[u].pop('contraction', None)  # networkx's copy of v's attributes
                G.nodes[u]['pos'] = new_pos
                G.nodes[u]['radius'] = new_radius
                G.nodes[u]['merged'] = True
                contracting = True
                stages['rung_contractions'] = stages.get('rung_contractions', 0) + 1
                break  # topology changed; restart the scan

    # -------------------------------------------------------------------------
    # 7. SMOOTHING
    # -------------------------------------------------------------------------
    print(f"  Smoothing ({smooth_iters_effective} iterations)...")
    G = laplacian_smooth_graph(G, iterations=smooth_iters_effective, alpha=smooth_alpha)

    # 7.5 RADIUS -- equivalent-cylinder radius on the final node positions.
    # The EDT value read at the skeleton voxel is kept as 'radius_edt'.
    print("  Estimating radii (equivalent cylinder)...")
    edt_r = {n: float(G.nodes[n]['radius']) for n in G.nodes()}
    eq_r = equivalent_cylinder_radii(G, mask_points_world, voxel_volume, edt_r)
    for n in G.nodes():
        G.nodes[n]['radius_edt'] = edt_r[n]
        G.nodes[n]['radius'] = eq_r[n]

    qc = graph_qc(G, mask_index, mask_points_world, voxel_volume, crop_shape, float(np.min(voxel_sizes)))
    qc.update(stages)
    qc['runtime_s'] = round(_time.time() - _t0, 1)
    del mask_points_world, mask_index

    # -------------------------------------------------------------------------
    # 8. SAVE OUTPUTS
    # -------------------------------------------------------------------------
    print("  Saving Results...")
    for u, v in G.edges():
        G[u][v]['weight'] = np.linalg.norm(G.nodes[u]['pos'] - G.nodes[v]['pos'])

    # Radii are also stored per-node; this flat map is what Metric_extractor reads.
    legacy_radius_map = {n: G.nodes[n]['radius'] for n in G.nodes()}

    package = {
        'graph': G,
        'node_radius_map': legacy_radius_map,
        'voxel_sizes': voxel_sizes,          # isotropic spacing stored here
        'input_file': nifti_path,
        'params': params,
        'code_version': code_version(),
        'qc': qc,
    }
    with open(output_pkl, 'wb') as f:
        pickle.dump(package, f)
    import json as _json
    with open(output_qc, 'w') as f:
        _json.dump({'input_file': nifti_path, 'code_version': package['code_version'], 'params': params, 'qc': qc},
                   f, indent=1, default=float)

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
                             "(default: 2); scaled per-scan by voxel spacing.")
    parser.add_argument("--smooth_alpha",   type=float, default=0.8,
                        help="Laplacian smoothing alpha (default: 0.8).")
    parser.add_argument("--smooth_reference_spacing", type=float, default=0.5,
                        help="Reference isotropic spacing (mm) that --smooth_iters is "
                             "calibrated for (default: 0.5).")
    parser.add_argument("--intensity_thresh", type=float, default=0.6,
                        help="Intensity threshold for edge pruning (default: 0.6).")
    parser.add_argument("--target_iso",     type=float, default=None,
                        help="Target isotropic voxel spacing in mm (default: auto -- this "
                             "image's finest native axis, so no axis is downsampled).")
    parser.add_argument("--no_extend_tips", action="store_true",
                        help="Do not extend skeleton tips that stop short of their vessel end "
                             "(extension is on by default).")
    parser.add_argument("--mask_smoothing", type=float, default=0.0,
                        help="Gaussian smoothing (mm) of the mask boundary before "
                             "skeletonisation, for noisy segmentations only (default: 0 = off; "
                             "0.25 validated; raised to at least 0.7 voxel; erodes vessels thinner "
                             "than ~1 voxel).")
    parser.add_argument("--min_branch_len",  type=float, default=2.0,
                        help="Prune terminal branches (leaf-to-bifurcation hairs) shorter than "
                             "this many mm as skeletonization spurs (default: 2.0); raise to "
                             "remove more hairs.")

    args = parser.parse_args()

    try:
        extract_graph(
            nifti_path=args.mask,
            output_folder=args.output_dir,
            smooth_iters=args.smooth_iters,
            smooth_alpha=args.smooth_alpha,
            smooth_reference_spacing_mm=args.smooth_reference_spacing,
            intensity_threshold=args.intensity_thresh,
            target_iso_spacing=args.target_iso,
            min_branch_length_mm=args.min_branch_len,
            mask_smoothing_mm=args.mask_smoothing,
            extend_tips=not args.no_extend_tips
        )
    except Exception as e:
        print(f"Error during graph extraction: {e}")
        raise