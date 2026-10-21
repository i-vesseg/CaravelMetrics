# -*- coding: utf-8 -*-
"""
Branch_metrics.py

Bifurcation angles on a Caravel graph (node attribute 'pos' in mm, a node
radius map in mm):

  bifurcation_metrics()   branching angles per junction

All lengths are geodesic polyline lengths along the graph. Validated on
phantoms with known geometry.
Principal vessel, taper, stenosis index and Murray exponent were tried and
removed (unstable on real masks or no study use; recoverable from commit
eb80845).
"""
import numpy as np

# Arc length (mm) excluded next to a junction when reading a branch's
# direction: the skeleton bends into bifurcation centres (same 2 mm as
# curvature).
END_EXCLUSION_MM = 2.0
# Window (mm, beyond the exclusion zone) used for a branch's direction and
# radius at a bifurcation.
BIF_WINDOW_MM = 4.0


def _chain_xyz(G, chain):
    return np.array([G.nodes[n]['pos'] for n in chain], dtype=float)


def _arclength(P):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


# ---------------------------------------------------------------------------
# bifurcations
# ---------------------------------------------------------------------------
def _branch_at(G, chain, rmap, r_junction):
    """Direction (unit vector pointing away from the junction) and radius of
    a branch, read in the window [d0, d0 + BIF_WINDOW_MM] of arc length from
    the junction, d0 = max(END_EXCLUSION_MM, junction radius): inside the
    junction the skeleton bends toward its centre."""
    P = _chain_xyz(G, chain)
    s = _arclength(P)
    d0 = max(END_EXCLUSION_MM, r_junction)
    d1 = d0 + BIF_WINDOW_MM
    if s[-1] < d0 + 1.0:
        return None, np.nan
    d1 = min(d1, s[-1])
    a = np.array([np.interp(d0, s, P[:, k]) for k in range(3)])
    b = np.array([np.interp(d1, s, P[:, k]) for k in range(3)])
    v = b - a
    nv = np.linalg.norm(v)
    if nv == 0:
        return None, np.nan
    r = np.array([rmap.get(n, np.nan) for n in chain], float)
    win = (s >= d0) & (s <= d1)
    rad = float(np.nanmedian(r[win])) if win.any() else np.nan
    return v / nv, rad


def _angle(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


PARENT_AMBIGUITY_DEG = 20.0


def _choose_parent(br):
    """Index of the parent among three (direction, radius) branches.

    Geometry first: the two daughters are the pair of branches with the
    smallest angle between them (they leave the junction on the same side);
    the third is the parent. If the two smallest pair angles are within
    PARENT_AMBIGUITY_DEG -- a side branch off a straight vessel (T-junction),
    where either half of the main vessel could be upstream -- the wider of
    the two parent candidates is taken. Choosing purely by radius (previous
    rule) picked the downstream half at T-junctions whenever it read
    slightly wider, producing 'bifurcation angles' of 150-170 deg."""
    pairs = sorted(((_angle(br[i][0], br[j][0]), i, j) for i, j in ((0, 1), (0, 2), (1, 2))))
    (a1, i1, j1), (a2, i2, j2) = pairs[0], pairs[1]
    p1 = ({0, 1, 2} - {i1, j1}).pop()
    if a2 - a1 >= PARENT_AMBIGUITY_DEG:
        return p1
    p2 = ({0, 1, 2} - {i2, j2}).pop()
    return max((p1, p2), key=lambda k: br[k][1])


# Daughters that rejoin within a loop of at most this perimeter (mm) make
# the junction a "short-loop junction": either a tunnel through one vessel
# (a hole in the mask that the skeleton goes around: the two "daughters" are
# the two sides of the same vessel, at 7-26 deg) or a tiny anastomosis. In
# neither case is the angle a branching angle. On the test masks this catches
# all six tunnels identified by eye.
SHORT_LOOP_MM = 25.0


def _daughters_rejoin(G, node, c1, c2, max_perimeter=SHORT_LOOP_MM):
    """True if the daughters leaving `node` along chains c1 and c2 meet again
    within a loop of perimeter <= max_perimeter (mm)."""
    import networkx as nx

    def w(a, b, _d):
        return float(np.linalg.norm(np.asarray(G.nodes[a]['pos'], float) - np.asarray(G.nodes[b]['pos'], float)))
    a, b = c1[1], c2[1]
    H = nx.restricted_view(G, [node], [])
    try:
        d = nx.dijkstra_path_length(H, a, b, weight=w)
    except nx.NetworkXNoPath:
        return False
    return d + w(node, a, None) + w(node, b, None) <= max_perimeter


def bifurcation_metrics(G, rmap, segments, junction_radius=None):
    """One record per junction of degree exactly 3 (degree >= 4 junctions are
    skipped: no unique parent/daughter split).

    The parent is chosen by geometry, then radius (_choose_parent). Returned
    angles (deg):
      bifurcation_angle   between the two daughters
      deflection_1/2      between the parent's continuation and each daughter
                          (1 = wider daughter)
    and on_short_loop (the daughters rejoin within SHORT_LOOP_MM; such
    junctions are kept in the table but left out of summary statistics).
    junction_radius: {node: radius} used to skip the junction core (default:
    the node radius map).
    """
    jr = junction_radius or rmap
    by_node = {}
    for c in segments:
        if c[0] == c[-1]:
            continue
        by_node.setdefault(c[0], []).append(list(c))
        by_node.setdefault(c[-1], []).append(list(reversed(c)))
    out = []
    for node, chains in by_node.items():
        if G.degree(node) != 3 or len(chains) != 3:
            continue
        rj = float(jr.get(node, np.nan)) if np.isfinite(jr.get(node, np.nan)) else 0.0
        br = [_branch_at(G, c, rmap, rj) for c in chains]
        if any(v is None or not np.isfinite(r) for v, r in br):
            continue
        p = _choose_parent(br)
        d = sorted([i for i in range(3) if i != p], key=lambda i: br[i][1], reverse=True)
        v0, v1, v2 = br[p][0], br[d[0]][0], br[d[1]][0]
        out.append(dict(node=node, pos=list(map(float, G.nodes[node]['pos'])),
                        bifurcation_angle=_angle(v1, v2),
                        deflection_1=_angle(-v0, v1), deflection_2=_angle(-v0, v2),
                        on_short_loop=bool(_daughters_rejoin(G, node, chains[d[0]], chains[d[1]]))))
    return out
