import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.ops import polygonize
from scipy.spatial import Voronoi, Delaunay



def make_bbox(xmin, xmax, ymin, ymax):
    """Return shapely Polygon bounding box."""
    return Polygon([(xmin, ymin), (xmin, ymax),
                    (xmax, ymax), (xmax, ymin)])


def voronoi_polygons_clipped(points, bbox_poly):
    """
    Build Voronoi diagram from points (Nx2), and clip each region to bbox.
    Returns list of shapely Polygons, one per point (same order).
    Unbounded cells are cropped to the bbox.
    """
    vor = Voronoi(points)
    polys = []

    for i, region_index in enumerate(vor.point_region):
        region_vertices = vor.regions[region_index]

        if -1 in region_vertices or len(region_vertices) == 0:

            finite_vertices = [v for v in region_vertices if v != -1]
            if len(finite_vertices) < 3:
                polys.append(None)
                continue
            poly_coords = [vor.vertices[v] for v in finite_vertices]
            poly = Polygon(poly_coords)
        else:
            poly_coords = [vor.vertices[v] for v in region_vertices]
            poly = Polygon(poly_coords)

        poly_clipped = poly.intersection(bbox_poly)

        if poly_clipped.is_empty:
            polys.append(None)
        else:
            if poly_clipped.geom_type == "MultiPolygon":
                poly_clipped = max(poly_clipped.geoms, key=lambda p: p.area)
            polys.append(poly_clipped)

    return polys


def lloyd_relaxation(points, bbox_poly, iterations=3):
    """
    Lloyd's algorithm:
    1. Build Voronoi
    2. Compute centroid of each (clipped) cell
    3. Move point to that centroid
    Repeat.

    This mimics cells pushing until cells are more evenly spaced.
    """
    pts = np.array(points, dtype=float)

    for _ in range(iterations):
        vor_polys = voronoi_polygons_clipped(pts, bbox_poly)

        new_pts = []
        for poly, old in zip(vor_polys, pts):
            if poly is None or poly.is_empty:
                # if something weird happens, keep old point
                new_pts.append(old)
            else:
                # centroid of the polygon
                cx, cy = poly.centroid.x, poly.centroid.y
                new_pts.append([cx, cy])

        pts = np.array(new_pts, dtype=float)

        # also, to be safe, clamp points into bbox
        minx, miny, maxx, maxy = bbox_poly.bounds
        pts[:, 0] = np.clip(pts[:, 0], minx, maxx)
        pts[:, 1] = np.clip(pts[:, 1], miny, maxy)

    return pts


def delaunay_edges(points):
    """
    Build Delaunay triangulation on points (Nx2).
    Return unique undirected edges as pairs of indices,
    and also as segments (2-point arrays).
    """
    tri = Delaunay(points)
    edges = set()

    for simplex in tri.simplices:
        i, j, k = simplex
        pairs = [(i, j), (j, k), (k, i)]
        for a, b in pairs:
            edge = tuple(sorted((a, b)))
            edges.add(edge)

    edges = list(edges)
    segments = [(points[i], points[j]) for i, j in edges]
    return edges, segments



def cell_neighbor_counts(edges, n_points):
    """
    Given undirected edges (i,j) between cells i and j,
    count how many neighbors each cell has.
    """
    counts = [0] * n_points
    for i, j in edges:
        counts[i] += 1
        counts[j] += 1
    return np.array(counts)


def polygon_areas(polys):
    """
    Return area for each polygon (or np.nan if missing).
    """
    areas = []
    for p in polys:
        if p is None or p.is_empty:
            areas.append(np.nan)
        else:
            areas.append(p.area)
    return np.array(areas)



def plot_cells_and_voronoi(points, polys, title="Voronoi Cells (Cell Territories)"):
    """
    Plot Voronoi polygons (colored by area) + nuclei points.
    """
    areas = polygon_areas(polys)
    # normalize areas just for color mapping
    # avoid NaN by replacing with mean
    finite_areas = areas[np.isfinite(areas)]
    if len(finite_areas) == 0:
        finite_areas = np.array([1.0])
    mean_area = finite_areas.mean()
    areas_for_color = np.where(np.isfinite(areas), areas, mean_area)

    plt.figure(figsize=(7,7))
    ax = plt.gca()

    # Draw polygons
    for poly, a in zip(polys, areas_for_color):
        if poly is None or poly.is_empty:
            continue
        x,y = poly.exterior.xy
        plt.fill(x, y, alpha=0.5,
                 # colormap by area: bigger area slightly darker
                 # (we'll just scale a into [0,1] manually)
                 color=plt.cm.viridis((a - areas_for_color.min()) /
                                      (areas_for_color.max() - areas_for_color.min() + 1e-9)),
                 edgecolor="black", linewidth=0.5)

    # Draw nuclei as black dots
    plt.scatter(points[:,0], points[:,1], c="black", s=15, zorder=10)

    ax.set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()


def plot_delaunay_network(points, segments, title="Delaunay Neighbor Graph"):
    """
    Draw Delaunay edges (cell adjacency) and nuclei.
    """
    plt.figure(figsize=(7,7))
    ax = plt.gca()

    # edges
    for seg in segments:
        p1, p2 = seg
        xs = [p1[0], p2[0]]
        ys = [p1[1], p2[1]]
        plt.plot(xs, ys, linewidth=0.5, color="gray")

    # nuclei
    plt.scatter(points[:,0], points[:,1], c="red", s=20, zorder=5)

    ax.set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()


def plot_histogram(data, xlabel, title):
    """
    Generic histogram plotter (areas, neighbor counts, etc.).
    """
    clean = data[np.isfinite(data)]
    plt.figure(figsize=(6,4))
    plt.hist(clean, bins=15, edgecolor="black", alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    N_CELLS = 100          # how many cells/nuclei
    BOX_MIN, BOX_MAX = 0, 1.0   # tissue square [0,1]x[0,1]
    LLOYD_ITERS = 5        # how many relaxation steps (0 = off)

    # 1. make bounding box polygon
    bbox_poly = make_bbox(BOX_MIN, BOX_MAX, BOX_MIN, BOX_MAX)

    # 2. generate random nuclei in that box
    rng = np.random.default_rng(seed=42)
    pts = rng.random((N_CELLS, 2)) * (BOX_MAX - BOX_MIN) + BOX_MIN
    # --- PLOT BEFORE RELAXATION (initial random configuration) ---
    vor_polys_before = voronoi_polygons_clipped(pts, bbox_poly)
    plot_cells_and_voronoi(
        pts,
        vor_polys_before,
        title="Cell Packing via Voronoi (Before Growth/Relaxation)"
    )

    # 3. relax (optional but looks nicer / more bio-like)
    pts = lloyd_relaxation(pts, bbox_poly, iterations=LLOYD_ITERS)

    # 4. final Voronoi cells (clipped to tissue boundary)
    vor_polys = voronoi_polygons_clipped(pts, bbox_poly)

    # 5. Delaunay neighbors
    edges, segments = delaunay_edges(pts)

    # 6. stats: area + neighbor counts
    areas = polygon_areas(vor_polys)          # "cell size"
    neigh_counts = cell_neighbor_counts(edges, len(pts))  # how many touching neighbors

    print("=== Cell area stats (units^2) ===")
    print(f"mean area: {np.nanmean(areas):.4f}")
    print(f"std area : {np.nanstd(areas):.4f}")
    print(f"min area : {np.nanmin(areas):.4f}")
    print(f"max area : {np.nanmax(areas):.4f}")
    print()

    print("=== Neighbor count stats ===")
    print(f"mean neighbors: {np.mean(neigh_counts):.2f}")
    print(f"std neighbors : {np.std(neigh_counts):.2f}")
    print(f"min neighbors : {np.min(neigh_counts)}")
    print(f"max neighbors : {np.max(neigh_counts)}")

    # 7. plots for your report
    plot_cells_and_voronoi(
        pts,
        vor_polys,
        title="Cell Packing via Voronoi (After Growth/Relaxation)"
    )

    plot_delaunay_network(
        pts,
        segments,
        title="Cell-Cell Adjacency via Delaunay Triangulation"
    )

    # histogram of areas (cell sizes)
    plot_histogram(
        areas,
        xlabel="Cell area",
        title="Distribution of Cell Areas"
    )

    # histogram of neighbor counts
    plot_histogram(
        neigh_counts,
        xlabel="Number of neighbors",
        title="Distribution of Neighbor Counts"
    )
