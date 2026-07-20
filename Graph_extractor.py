import os
import argparse
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

def extract_graph(
    nifti_path: str, 
    output_folder: str,
    smooth_iters: int = 2,
    smooth_alpha: float = 0.8,
    intensity_threshold: float = 0.6,
    min_component_size: int = 7,
    orphan_distance_threshold: float = 2.5,
    merge_distance: float = 3.0,
    min_angle: float = 30.0
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
    
    # 1. LOAD DATA
    print("  Loading NIFTI...")
    nii = nib.load(nifti_path)
    data = nii.get_fdata()
    affine = nii.affine
    inv_affine = np.linalg.inv(affine)
    voxel_sizes = nib.affines.voxel_sizes(affine)

    # 2. MESH GENERATION
    print("  Generating Reference Mesh...")
    vol = vedo.Volume(data)
    mesh = vol.isosurface(value=0.5).apply_transform(affine.tolist())
    mesh.clean()

    # 3. SKELETONIZATION
    print("  Skeletonizing...")
    skeleton_mask = skeletonize(data > 0)
    skel_indices = np.argwhere(skeleton_mask)
    skel_points_world = apply_affine(affine, skel_indices)
    
    # Generate Distance Transform for accurate radius sampling
    full_dist_map_vox = distance_transform_edt(data > 0, sampling=voxel_sizes)

    # 4. ORPHAN HANDLING 
    print(f"  Handling Orphans (Threshold > {orphan_distance_threshold})...")
    dist_to_skel = distance_transform_edt(~skeleton_mask, sampling=voxel_sizes)
    orphan_mask = (data > 0) & (dist_to_skel > orphan_distance_threshold)
    labeled_orphans, num_zones = label(orphan_mask)

    orphan_points = []
    for z in range(1, num_zones + 1):
        cm = center_of_mass(orphan_mask, labeled_orphans, z)
        orphan_points.append(apply_affine(affine, np.array(cm)))
    orphan_points_world = np.array(orphan_points)

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
        print(f"    Orphans kept after merge: {len(orphan_points_world)}")

    # 5. BUILD GRAPH (AND EXTRACT ACCURATE RADIUS)
    print("  Building Graph and Extracting Native Radii...")
    G = nx.Graph()
    
    for i, pt in enumerate(skel_points_world):
        pos_vox = apply_affine(inv_affine, pt).astype(int)
        pos_vox = np.clip(pos_vox, 0, np.array(data.shape) - 1)
        r = full_dist_map_vox[tuple(pos_vox)]
        G.add_node(i, pos=pt, radius=r)

    offset = len(skel_points_world)
    for i, pt in enumerate(orphan_points_world):
        pos_vox = apply_affine(inv_affine, pt).astype(int)
        pos_vox = np.clip(pos_vox, 0, np.array(data.shape) - 1)
        r = full_dist_map_vox[tuple(pos_vox)]
        G.add_node(offset + i, pos=pt, radius=r)

    idx_map = {tuple(p): i for i, p in enumerate(skel_indices)}
    offsets_26 = [np.array([z, y, x]) for z in (-1, 0, 1) for y in (-1, 0, 1) for x in (-1, 0, 1) if
                  not (z == 0 and y == 0 and x == 0)]
    for i, curr in enumerate(skel_indices):
        for off in offsets_26:
            neigh = tuple(curr + off)
            if neigh in idx_map and idx_map[neigh] > i:
                G.add_edge(i, idx_map[neigh])

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

            is_bridge = False
            if len(valid_candidates) > 1:
                for cand_a, cand_b in itertools.combinations(valid_candidates, 2):
                    angle = calculate_angle(opos, cand_a['pos'], cand_b['pos'])
                    if angle > min_angle:
                        is_bridge = True
                        break

            if is_bridge:
                nodes_to_remove.append(oid)
            elif len(valid_candidates) > 0:
                valid_candidates.sort(key=lambda x: x['dist'])
                best_idx = valid_candidates[0]['idx']
                G.add_edge(oid, best_idx)
            else:
                nodes_to_remove.append(oid)

        G.remove_nodes_from(nodes_to_remove)

    # 6. PRUNING
    print("  Pruning...")
    for tri in [c for c in nx.enumerate_all_cliques(G) if len(c) == 3]:
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        longest = max(edges, key=lambda e: np.linalg.norm(G.nodes[e[0]]['pos'] - G.nodes[e[1]]['pos']) if G.has_edge(
            *e) else -1)
        if G.has_edge(*longest): G.remove_edge(*longest)

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
                    if (geo_dist / dist) > 2.0: edges_to_remove.append((u, v))
                else:
                    edges_to_remove.append((u, v))
            except:
                edges_to_remove.append((u, v))
    G.remove_edges_from(edges_to_remove)

    # 6.5 FILTER SMALL COMPONENTS (Physically Agnostic)
    min_component_length_mm = 3.5  # E.g., drop anything shorter than 3.5 mm
    print(f"  Filtering Components < {min_component_length_mm} mm...")
    
    nodes_to_remove = []
    for component in nx.connected_components(G):
        sub_G = G.subgraph(component)
        comp_length = 0.0
        
        # Calculate true physical length
        for u, v in sub_G.edges():
            p1 = G.nodes[u]['pos']
            p2 = G.nodes[v]['pos']
            comp_length += np.linalg.norm(p1 - p2)
            
        if comp_length < min_component_length_mm:
            nodes_to_remove.extend(list(component))
            
    G.remove_nodes_from(nodes_to_remove)

    # 6.6 PRUNE FALSE BRANCHES (Hairs)
    min_branch_length_mm = 1.5
    print(f"  Pruning terminal branches < {min_branch_length_mm} mm...")
    
    pruning = True
    while pruning:
        pruning = False
        endpoints = [n for n, d in G.degree() if d == 1]
        for ep in endpoints:
            neighbor = list(G.neighbors(ep))[0]
            # If the neighbor is a bifurcation, measure the distance
            if G.degree(neighbor) >= 3:
                dist = np.linalg.norm(G.nodes[ep]['pos'] - G.nodes[neighbor]['pos'])
                if dist < min_branch_length_mm:
                    G.remove_node(ep)
                    pruning = True  # Re-check in case we created new endpoints


    # 7. SMOOTHING (Dynamically scaled to physical units)
    target_smoothing_mm = 1.5  # Physical distance we want the smoothing to "reach"
    mean_voxel = np.mean(voxel_sizes)
    dynamic_iters = max(1, int(round(target_smoothing_mm / mean_voxel)))
    
    print(f"  Smoothing ({dynamic_iters} iterations based on {mean_voxel:.2f}mm voxels)...")
    G = laplacian_smooth_graph(G, iterations=dynamic_iters, alpha=smooth_alpha)
    # 8. SAVE OUTPUTS AND SNAPSHOTS
    print("  Saving Results and Snapshots...")
    for u, v in G.edges():
        G[u][v]['weight'] = np.linalg.norm(G.nodes[u]['pos'] - G.nodes[v]['pos'])

    # Temporary legacy map to prevent breaking Metric_extractor until we rewrite it
    legacy_radius_map = {n: G.nodes[n]['radius'] for n in G.nodes()}

    package = {
        'graph': G,
        'node_radius_map': legacy_radius_map, 
        'voxel_sizes': voxel_sizes,
        'input_file': nifti_path
    }
    with open(output_pkl, 'wb') as f:
        pickle.dump(package, f)

    # --- EXPORT VTP FILES ---
    s_pts = [G.nodes[u]['pos'] for u, v in G.edges()]
    e_pts = [G.nodes[v]['pos'] for u, v in G.edges()]
    
    if s_pts:
        # 1. Native VTP (For ParaView, ITK-SNAP, standard VTK)
        lines_native = vedo.Lines(s_pts, e_pts).c('green').lw(3)
        lines_native.write(output_vtp)
        print(f"    -> Saved PKL: {output_pkl}")
        print(f"    -> Saved Native VTP: {output_vtp}")
        
        # 2. Slicer-Specific VTP (Applies LPS-to-RAS flip)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vessel Graph Extractor Module")
    
    parser.add_argument("--mask", required=True, help="Path to the binary vessel mask NIfTI file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the resulting .pkl and .vtp files.")
    
    parser.add_argument("--smooth_iters", type=int, default=2, help="Iterations for Laplacian smoothing (default: 2).")
    parser.add_argument("--smooth_alpha", type=float, default=0.8, help="Alpha value for Laplacian smoothing (default: 0.8).")
    parser.add_argument("--intensity_thresh", type=float, default=0.6, help="Intensity threshold for edge pruning (default: 0.6).")
    parser.add_argument("--min_comp_size", type=int, default=7, help="Minimum number of nodes to keep a component (default: 7).")
    parser.add_argument("--orphan_dist", type=float, default=2.5, help="Distance threshold for orphan detection (default: 2.5).")
    parser.add_argument("--merge_dist", type=float, default=3.0, help="Distance threshold for merging orphans (default: 3.0).")
    parser.add_argument("--min_angle", type=float, default=30.0, help="Minimum angle (degrees) to classify as a bridge (default: 30.0).")

    args = parser.parse_args()

    try:
        extract_graph(
            nifti_path=args.mask,
            output_folder=args.output_dir,
            smooth_iters=args.smooth_iters,
            smooth_alpha=args.smooth_alpha,
            intensity_threshold=args.intensity_thresh,
            min_component_size=args.min_comp_size,
            orphan_distance_threshold=args.orphan_dist,
            merge_distance=args.merge_dist,
            min_angle=args.min_angle
        )
    except Exception as e:
        print(f"Error during graph extraction: {e}")