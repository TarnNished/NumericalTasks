
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
from scipy.interpolate import CubicSpline


# ----------------------------------------------------------
# 1. Utility functions
# ----------------------------------------------------------

def load_image(path):
    """Load image in BGR (OpenCV) and also return grayscale."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def detect_edges(gray, blur_ksize=5, canny_low=50, canny_high=150):
    """Simple Canny edge detector with Gaussian blur."""
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    return edges


def extract_profile(edges, axis_x=None, min_radius_px=1):
    """
    Extract right-side profile of the bulb.

    For each row y, look to the right of axis_x and find the first edge pixel.
    Radius = (x_edge - axis_x). We use only rows where we find an edge.

    Returns:
        z: vertical coordinate (in pixels, bottom = 0, top > 0)
        r: radius (in pixels)
    """
    h, w = edges.shape

    # choose symmetry axis: by default vertical center
    if axis_x is None:
        axis_x = w // 2

    z_list = []
    r_list = []

    # we scan from bottom (h-1) to top (0) to make z=0 at bottom
    for y in range(h - 1, -1, -1):
        row = edges[y, :]
        # look for edge pixels to the right of axis_x
        right_part = row[axis_x:]
        xs = np.where(right_part > 0)[0]
        if xs.size == 0:
            continue

        # furthest right edge pixel (outer boundary)
        x_edge = axis_x + xs[-1]
        radius = x_edge - axis_x

        if radius >= min_radius_px:
            z_list.append(h - 1 - y)  # z = 0 at bottom
            r_list.append(radius)

    z = np.array(z_list, dtype=float)
    r = np.array(r_list, dtype=float)

    # sort by z just in case
    idx = np.argsort(z)
    z = z[idx]
    r = r[idx]

    return z, r


def fit_radius_spline(z, r):
    """
    Fit a natural cubic spline r(z).

    We also remove duplicate z values if any.
    """
    z_unique, idx_unique = np.unique(z, return_index=True)
    r_unique = r[idx_unique]

    spline = CubicSpline(z_unique, r_unique, bc_type='natural')
    return spline, z_unique.min(), z_unique.max()


def compute_volume(spline, z_min, z_max, num_samples=500):
    """
    Compute volume of solid of revolution obtained by rotating r(z)
    around the z-axis:

        V = ∫ π r(z)^2 dz from z_min to z_max
    """
    z_dense = np.linspace(z_min, z_max, num_samples)
    r_dense = spline(z_dense)
    # enforce non-negative radius
    r_dense = np.maximum(r_dense, 0.0)
    volume = np.trapz(np.pi * r_dense**2, z_dense)
    return volume, z_dense, r_dense


def create_3d_surface(z_dense, r_dense, num_theta=60):
    """
    Create 3D surface coordinates by revolving r(z) around z-axis.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, num_theta)
    Z, Theta = np.meshgrid(z_dense, theta)

    # r_dense is (N,), tile to shape (num_theta, N)
    R = np.tile(r_dense, (num_theta, 1))

    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    return X, Y, Z


# ----------------------------------------------------------
# 2. Plotting helpers
# ----------------------------------------------------------

def show_edges(img_bgr, edges):
    """Show original image and edges side by side."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Canny edges")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_profile_and_spline(z, r, spline, z_dense, r_dense):
    """Plot extracted nodes and fitted spline profile."""
    plt.figure(figsize=(5, 6))
    plt.scatter(r, z, label="extracted points", s=10)
    plt.plot(r_dense, z_dense, 'r-', label="cubic spline")
    plt.gca().invert_yaxis()  # optional, if you want top at smaller z visually
    plt.xlabel("radius (pixels)")
    plt.ylabel("height z (pixels)")
    plt.title("Thermometer bulb profile (2D)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_3d_surface(X, Y, Z):
    """3D surface plot of reconstructed bulb."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D reconstructed thermometer bulb")
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# 3. Main pipeline
# ----------------------------------------------------------

def main():
    image_path = "thermometer.jpg"

    # 1) Load image and detect edges
    img, gray = load_image(image_path)
    edges = detect_edges(gray, blur_ksize=5, canny_low=50, canny_high=150)
    show_edges(img, edges)

    # 2) Extract profile (one side) relative to vertical axis
    h, w = edges.shape
    axis_x = w // 2

    z, r = extract_profile(edges, axis_x=axis_x, min_radius_px=2)

    if len(z) < 5:
        print("Not enough profile points found. Try adjusting Canny thresholds or cropping image.")
        return

    # 3) Fit spline r(z)
    spline, z_min, z_max = fit_radius_spline(z, r)

    # 4) Compute volume
    volume, z_dense, r_dense = compute_volume(spline, z_min, z_max, num_samples=600)
    print(f"Estimated bulb volume (in pixel^3 units): {volume:.2f}")

    # 5) Visualize 2D profile + spline
    show_profile_and_spline(z, r, spline, z_dense, r_dense)

    # 6) Build 3D surface and visualize
    X, Y, Z = create_3d_surface(z_dense, r_dense, num_theta=80)
    show_3d_surface(X, Y, Z)


if __name__ == "__main__":
    main()
