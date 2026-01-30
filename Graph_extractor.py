import os
import glob
import pickle
import vedo
import nibabel as nib
import numpy as np
import networkx as nx
import itertools
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, label, center_of_mass, map_coordinates
from scipy.spatial import cKDTree
from nibabel.affines import apply_affine

# =========================================================================
#                               CONFIGURATION
# =========================================================================

# 3. PARAMETERS
SMOOTH_ITERS = 2
SMOOTH_ALPHA = 0.8
INTENSITY_THRESHOLD = 0.6
MIN_COMPONENT_SIZE = 7

# 4. ORPHAN SETTINGS
ORPHAN_DISTANCE_THRESHOLD = 2.5
MERGE_DISTANCE = 3.0
MIN_ANGLE = 30.0

# 5. ALIGNMENT MATRIX
SLICER_MATRIX = np.array([
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])


# =========================================================================
#                             HELPER FUNCTIONS
# =========================================================================

def calculate_angle(p_center, p1, p2):
    """Calculates angle (degrees) formed by p1-center-p2."""
    v1 = p1 - p_center
    v2 = p2 - p_center
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    dot = np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def laplacian_smooth_graph(graph, iterations, alpha):
    """Smooths node positions using Laplacian smoothing."""
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


def extract_graph(nifti_path, output_folder):
    """
    Runs the full graph extraction and alignment pipeline for a single file.
    """
    case_name = os.path.basename(output_folder)
    print(f"\n{'=' * 60}")
    print(f"PROCESSING CASE: {case_name}")
    print(f"File: {nifti_path}")
    print(f"{'=' * 60}")

    # Define output filenames
    output_pkl = os.path.join(output_folder, "vessel_data.pkl")
    output_vtp = os.path.join(output_folder, "vessel_graph.vtp")
    
    if os.path.exists(output_pkl):
        print(f"  -> Output PKL already exists. Skipping case.")
        return
    
    # 1. LOAD DATA
    print("  Loading NIFTI...")
    nii = nib.load(nifti_path)
    data = nii.get_fdata()
    affine = nii.affine
    inv_affine = np.linalg.inv(affine)
    voxel_sizes = nib.affines.voxel_sizes(affine)
    mean_voxel_size = np.mean(voxel_sizes)

    # 2. MESH GENERATION (Reference for Geodesic)
    print("  Generating Reference Mesh...")
    vol = vedo.Volume(data)
    mesh = vol.isosurface(value=0.5).apply_transform(affine)
    mesh.clean()

    # 3. SKELETONIZATION
    print("  Skeletonizing...")
    skeleton_mask = skeletonize(data > 0)
    skel_indices = np.argwhere(skeleton_mask)
    skel_points_world = apply_affine(affine, skel_indices)
    full_dist_map_vox = distance_transform_edt(data > 0)

    # 4. ORPHAN HANDLING (UPDATED LOGIC)
    print(f"  Handling Orphans (Threshold > {ORPHAN_DISTANCE_THRESHOLD})...")
    dist_to_skel = distance_transform_edt(~skeleton_mask, sampling=voxel_sizes)
    orphan_mask = (data > 0) & (dist_to_skel > ORPHAN_DISTANCE_THRESHOLD)
    labeled_orphans, num_zones = label(orphan_mask)

    orphan_points = []
    for z in range(1, num_zones + 1):
        cm = center_of_mass(orphan_mask, labeled_orphans, z)
        orphan_points.append(apply_affine(affine, np.array(cm)))
    orphan_points_world = np.array(orphan_points)

    # Cluster & Merge Logic
    if len(orphan_points_world) > 0:
        tree = cKDTree(orphan_points_world)
        pairs = tree.query_pairs(r=MERGE_DISTANCE)
        g_tmp = nx.Graph()
        g_tmp.add_nodes_from(range(len(orphan_points_world)))
        g_tmp.add_edges_from(pairs)

        merged_indices = []
        for component in nx.connected_components(g_tmp):
            comp_list = list(component)
            cluster_coords = orphan_points_world[comp_list]
            centroid = np.mean(cluster_coords, axis=0)
            # Find closest orphan to centroid
            closest_idx = np.argmin(np.linalg.norm(cluster_coords - centroid, axis=1))
            merged_indices.append(comp_list[closest_idx])

        if merged_indices:
            orphan_points_world = orphan_points_world[merged_indices]
        else:
            orphan_points_world = np.array([])
        print(f"    Orphans kept after merge: {len(orphan_points_world)}")

    # 5. BUILD GRAPH
    print("  Building Graph...")
    G = nx.Graph()
    for i, pt in enumerate(skel_points_world):
        G.add_node(i, pos=pt)

    offset = len(skel_points_world)
    for i, pt in enumerate(orphan_points_world):
        G.add_node(offset + i, pos=pt)

    # Connect Skeleton
    idx_map = {tuple(p): i for i, p in enumerate(skel_indices)}
    offsets_26 = [np.array([z, y, x]) for z in (-1, 0, 1) for y in (-1, 0, 1) for x in (-1, 0, 1) if
                  not (z == 0 and y == 0 and x == 0)]
    for i, curr in enumerate(skel_indices):
        for off in offsets_26:
            neigh = tuple(curr + off)
            if neigh in idx_map and idx_map[neigh] > i:
                G.add_edge(i, idx_map[neigh])

    # Connect Orphans (UPDATED: Delete Bridges, Keep Endpoints)
    if len(orphan_points_world) > 0:
        skel_tree = cKDTree(skel_points_world)
        nodes_to_remove = []

        for i, opos in enumerate(orphan_points_world):
            oid = offset + i
            dists, idxs = skel_tree.query(opos, k=5)

            valid_candidates = []
            try:
                osurf = mesh.closest_point(opos, return_point_id=True)
                for j, c_idx in enumerate(idxs):
                    if dists[j] == float('inf'): continue
                    csurf = mesh.closest_point(skel_points_world[c_idx], return_point_id=True)
                    path = mesh.geodesic(osurf, csurf)
                    if path and len(path.vertices) > 0:
                        geo_len = np.sum(np.linalg.norm(np.diff(path.vertices, axis=0), axis=1))
                        valid_candidates.append({'dist': geo_len, 'idx': c_idx, 'pos': skel_points_world[c_idx]})
            except:
                pass

            # --- DECISION LOGIC ---
            is_bridge = False

            if len(valid_candidates) > 1:
                # Check for wide angle (> 30 degrees)
                for cand_a, cand_b in itertools.combinations(valid_candidates, 2):
                    angle = calculate_angle(opos, cand_a['pos'], cand_b['pos'])
                    if angle > MIN_ANGLE:
                        is_bridge = True
                        break

            if is_bridge:
                # Bridge -> DELETE
                nodes_to_remove.append(oid)
            elif len(valid_candidates) > 0:
                # Endpoint -> KEEP
                valid_candidates.sort(key=lambda x: x['dist'])
                best_idx = valid_candidates[0]['idx']
                G.add_edge(oid, best_idx)
            else:
                # Unconnected -> DELETE
                nodes_to_remove.append(oid)

        G.remove_nodes_from(nodes_to_remove)
        print(f"    Orphans removed (bridges/unconnected): {len(nodes_to_remove)}")

    # 6. PRUNING
    print("  Pruning...")
    # Triangles
    for tri in [c for c in nx.enumerate_all_cliques(G) if len(c) == 3]:
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        longest = max(edges, key=lambda e: np.linalg.norm(G.nodes[e[0]]['pos'] - G.nodes[e[1]]['pos']) if G.has_edge(
            *e) else -1)
        if G.has_edge(*longest): G.remove_edge(*longest)

    # Intensity & Geodesic
    step = np.min(voxel_sizes) * 0.5
    edges_to_remove = []
    for u, v in G.edges():
        p1, p2 = G.nodes[u]['pos'], G.nodes[v]['pos']
        dist = np.linalg.norm(p2 - p1)
        pts = p1 + np.outer(np.linspace(0, 1, max(2, int(dist / step) + 1)), p2 - p1)
        vals = map_coordinates(data, apply_affine(inv_affine, pts).T, order=1)
        if np.any(vals < INTENSITY_THRESHOLD):
            try:
                id1 = mesh.closest_point(p1, return_point_id=True)
                id2 = mesh.closest_point(p2, return_point_id=True)
                path_obj = mesh.geodesic(id1, id2)
                if path_obj:
                    geo_dist = np.sum(np.linalg.norm(np.diff(path_obj.vertices, axis=0), axis=1))
                    if (geo_dist / dist) > 2.0: edges_to_remove.append((u, v))
                else:
                    edges_to_remove.append((u, v))
            except:
                edges_to_remove.append((u, v))
    G.remove_edges_from(edges_to_remove)

    # 6.5 FILTER SMALL COMPONENTS (NEW)
    print(f"  Filtering Components < {MIN_COMPONENT_SIZE} nodes...")
    initial_node_count = G.number_of_nodes()
    nodes_to_remove = []
    for component in nx.connected_components(G):
        if len(component) < MIN_COMPONENT_SIZE:
            nodes_to_remove.extend(list(component))
    G.remove_nodes_from(nodes_to_remove)
    print(f"    Removed {len(nodes_to_remove)} nodes. (Graph size: {initial_node_count} -> {G.number_of_nodes()})")

    # 7. SMOOTHING
    print("  Smoothing...")
    G = laplacian_smooth_graph(G, iterations=SMOOTH_ITERS, alpha=SMOOTH_ALPHA)

    # 8. APPLY ALIGNMENT (SLICER MATRIX)
    print("  Applying Alignment Matrix...")
    rotation = SLICER_MATRIX[:3, :3]
    translation = SLICER_MATRIX[:3, 3]

    for n in G.nodes():
        pos = G.nodes[n]['pos']
        new_pos = np.dot(rotation, pos) + translation
        G.nodes[n]['pos'] = new_pos

    # 9. SAVE OUTPUTS
    print("  Saving Results...")
    node_radius_map = {}
    inv_slicer_rot = np.linalg.inv(rotation)

    # Calculate radii (mapping back to original space)
    for n in G.nodes():
        aligned_pos = G.nodes[n]['pos']
        orig_world_pos = np.dot(inv_slicer_rot, (aligned_pos - translation))
        pos_vox = apply_affine(inv_affine, orig_world_pos).astype(int)
        pos_vox = np.clip(pos_vox, 0, np.array(data.shape) - 1)
        node_radius_map[n] = full_dist_map_vox[tuple(pos_vox)] * mean_voxel_size

    # Update Weights
    for u, v in G.edges():
        G[u][v]['weight'] = np.linalg.norm(G.nodes[u]['pos'] - G.nodes[v]['pos'])

    # Save PKL
    package = {
        'graph': G,
        'node_radius_map': node_radius_map,
        'voxel_sizes': voxel_sizes,
        'input_file': nifti_path
    }
    with open(output_pkl, 'wb') as f:
        pickle.dump(package, f)

    # Save VTP
    s_pts = [G.nodes[u]['pos'] for u, v in G.edges()]
    e_pts = [G.nodes[v]['pos'] for u, v in G.edges()]
    if s_pts:
        lines = vedo.Lines(s_pts, e_pts).c('green').lw(3)
        lines.write(output_vtp)
        print(f"    -> Saved PKL: {output_pkl}")
        print(f"    -> Saved VTP: {output_vtp}")
    else:
        print("    ! Warning: Graph is empty, no VTP saved.")

    print(f"  Done with {case_name}.\n")
