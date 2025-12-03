

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, splprep, splev


# ==========================================================
# 1. Utility: parametrization and basic helpers
# ==========================================================

def parameterize(points: np.ndarray) -> np.ndarray:
    """
    Compute parameter t in [0, 1] based on cumulative arc-length.

    points: array of shape (N, 2)
    returns: t of shape (N,)
    """
    diffs = np.diff(points, axis=0)
    seg_len = np.sqrt((diffs ** 2).sum(axis=1))
    t = np.concatenate([[0.0], np.cumsum(seg_len)])
    if t[-1] == 0:
        return t  # degenerate case
    t /= t[-1]
    return t


def downsample_points(points: np.ndarray, step: int) -> np.ndarray:
    """
    Keep every 'step'-th point to simulate using fewer nodes.
    Always keeps the last point to close the shape nicely.
    """
    if step <= 1 or len(points) <= 2:
        return points
    reduced = points[::step]
    if not np.allclose(reduced[-1], points[-1]):
        reduced = np.vstack([reduced, points[-1]])
    return reduced


# ==========================================================
# 2. Spline fitting functions
# ==========================================================

def fit_natural_cubic(points: np.ndarray):
    """
    Fit natural cubic splines x(t), y(t) through all nodes.

    returns: spline_x(t), spline_y(t) as callable objects
    """
    t = parameterize(points)
    x, y = points[:, 0], points[:, 1]
    spline_x = CubicSpline(t, x, bc_type="natural")
    spline_y = CubicSpline(t, y, bc_type="natural")
    return spline_x, spline_y


def fit_bspline(points: np.ndarray, s: float = 0.0, k: int = 3):
    """
    Fit a parametric cubic B-spline using scipy.interpolate.splprep.

    s: smoothing factor (0 = interpolate, >0 = smoother, does not pass through all nodes)
    k: spline degree (3 = cubic)
    returns: tck (spline representation), u (internal parameter)
    """
    x, y = points[:, 0], points[:, 1]
    # splprep chooses its own parameter u in [0,1] by default
    tck, u = splprep([x, y], s=s, k=k)
    return tck, u


def eval_natural_cubic(points: np.ndarray, num_samples: int = 400):
    """
    Evaluate natural cubic spline at dense grid of t in [0, 1].
    """
    spline_x, spline_y = fit_natural_cubic(points)
    t_dense = np.linspace(0.0, 1.0, num_samples)
    x_dense = spline_x(t_dense)
    y_dense = spline_y(t_dense)
    return x_dense, y_dense


def eval_bspline(points: np.ndarray, s: float = 0.0, num_samples: int = 400):
    """
    Evaluate B-spline at dense grid of u in [0, 1].
    """
    tck, u = fit_bspline(points, s=s, k=3)
    u_dense = np.linspace(0.0, 1.0, num_samples)
    x_dense, y_dense = splev(u_dense, tck)
    return x_dense, y_dense


def approximation_error(points: np.ndarray,
                        spline_type: str = "natural",
                        s: float = 0.0) -> float:
    """
    Simple approximation error: MSE between original points and spline curve
    evaluated at (approximately) corresponding parameters.

    spline_type: "natural" or "bspline"
    s: smoothing factor for B-spline
    """
    x_true, y_true = points[:, 0], points[:, 1]
    n = len(points)

    if spline_type == "natural":
        t = parameterize(points)
        sx, sy = fit_natural_cubic(points)
        x_pred = sx(t)
        y_pred = sy(t)
    elif spline_type == "bspline":
        # evaluate B-spline at n approx equally spaced points
        tck, u = fit_bspline(points, s=s, k=3)
        u_dense = np.linspace(0.0, 1.0, n)
        x_pred, y_pred = splev(u_dense, tck)
    else:
        raise ValueError("Unknown spline_type, use 'natural' or 'bspline'.")

    mse = np.mean((x_true - x_pred) ** 2 + (y_true - y_pred) ** 2)
    return mse


# ==========================================================
# 3. Define sample points for Greek letters: α, θ, φ
# NOTE: Shapes are approximate; they just need to look like letters.
# ==========================================================

def letter_alpha_points() -> np.ndarray:
    """
    Approximate lowercase alpha (α) as a tilted loop with a small tail.
    """
    pts = np.array([
        [-1.0,  0.0],
        [-0.8,  0.6],
        [-0.3,  1.0],
        [ 0.3,  1.0],
        [ 0.8,  0.6],
        [ 1.0,  0.0],
        [ 0.8, -0.6],
        [ 0.3, -1.0],
        [-0.3, -1.0],
        [-0.8, -0.6],
        [-1.0,  0.0],    # close loop
        [ 0.2,  0.0],
        [ 0.8, -0.4]     # tail to the right
    ])
    return pts


def letter_theta_points() -> np.ndarray:
    """
    Approximate lowercase theta (θ) as an oval with a horizontal bar.
    """
    # outer oval
    outer = []
    for angle in np.linspace(0, 2 * np.pi, 20, endpoint=False):
        x = 0.9 * np.cos(angle)
        y = 1.2 * np.sin(angle)
        outer.append([x, y])
    outer = np.array(outer)

    # close the oval
    outer = np.vstack([outer, outer[0]])

    # add a horizontal bar inside
    bar = np.array([
        [-0.5, 0.0],
        [ 0.5, 0.0]
    ])

    pts = np.vstack([outer, bar])
    return pts


def letter_phi_points() -> np.ndarray:
    """
    Approximate lowercase phi (φ) as a vertical stem with an oval around it.
    """
    # vertical stem
    stem = np.array([
        [0.0, -1.3],
        [0.0, -0.5],
        [0.0,  0.5],
        [0.0,  1.3]
    ])

    # oval around center
    oval = []
    for angle in np.linspace(0, 2 * np.pi, 20, endpoint=False):
        x = 0.7 * np.cos(angle)
        y = 0.9 * np.sin(angle)
        oval.append([x, y])
    oval = np.array(oval)
    oval = np.vstack([oval, oval[0]])  # close oval

    pts = np.vstack([stem, oval])
    return pts


# ==========================================================
# 4. Plotting and experiments
# ==========================================================

def plot_letter_basic(points: np.ndarray, label: str):
    """
    Plot original nodes + natural cubic spline + B-spline (s=0)
    for one letter.
    """
    x_nc, y_nc = eval_natural_cubic(points)
    x_bs, y_bs = eval_bspline(points, s=0.0)

    plt.figure(figsize=(5, 5))
    plt.scatter(points[:, 0], points[:, 1],
                marker='o', label='nodes')
    plt.plot(x_nc, y_nc, label='natural cubic spline')
    plt.plot(x_bs, y_bs, '--', label='cubic B-spline (s=0)')
    plt.gca().set_aspect('equal', 'box')
    plt.title(f"Letter {label}: basic reconstruction")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def experiment_node_counts(points: np.ndarray, label: str):
    """
    Show effect of using fewer nodes for both spline types.
    """
    steps = [1, 2, 3]  # keep all points, every 2nd, every 3rd

    plt.figure(figsize=(12, 4))

    for i, step in enumerate(steps, start=1):
        pts_red = downsample_points(points, step)

        x_nc, y_nc = eval_natural_cubic(pts_red)
        x_bs, y_bs = eval_bspline(pts_red, s=0.0)

        mse_nc = approximation_error(pts_red, "natural")
        mse_bs = approximation_error(pts_red, "bspline")

        plt.subplot(1, len(steps), i)
        plt.scatter(pts_red[:, 0], pts_red[:, 1],
                    marker='o', label=f'nodes (step={step})')
        plt.plot(x_nc, y_nc, label=f'natural (MSE={mse_nc:.3e})')
        plt.plot(x_bs, y_bs, '--', label=f'B-spline (MSE={mse_bs:.3e})')
        plt.gca().set_aspect('equal', 'box')
        plt.title(f"{label}, step={step}")
        plt.grid(True)
        plt.legend(fontsize=8)

    plt.suptitle(f"Effect of node count on fitting quality – {label}")
    plt.tight_layout()
    plt.show()


def experiment_smoothing(points: np.ndarray, label: str):
    """
    Show effect of different B-spline smoothing parameters s.
    """
    smoothing_values = [0.0, 0.1, 0.5]

    plt.figure(figsize=(12, 4))

    for i, s in enumerate(smoothing_values, start=1):
        x_bs, y_bs = eval_bspline(points, s=s)
        mse_bs = approximation_error(points, "bspline", s=s)

        plt.subplot(1, len(smoothing_values), i)
        plt.scatter(points[:, 0], points[:, 1],
                    marker='o', label='nodes')
        plt.plot(x_bs, y_bs, '--',
                 label=f'B-spline s={s} (MSE={mse_bs:.3e})')
        plt.gca().set_aspect('equal', 'box')
        plt.title(f"{label}, s={s}")
        plt.grid(True)
        plt.legend(fontsize=8)

    plt.suptitle(f"Effect of B-spline smoothing on fitting – {label}")
    plt.tight_layout()
    plt.show()




def main():
    # Prepare points for all three Greek letters
    letters = {
        "α": letter_alpha_points(),
        "θ": letter_theta_points(),
        "φ": letter_phi_points(),
    }

    for label, pts in letters.items():
        print(f"\n=== Letter {label} ===")
        print(f"Number of raw nodes: {len(pts)}")

        mse_nat = approximation_error(pts, "natural")
        mse_bs0 = approximation_error(pts, "bspline", s=0.0)
        print(f"Natural cubic MSE:      {mse_nat:.3e}")
        print(f"B-spline (s=0) MSE:     {mse_bs0:.3e}")

        # Basic visualization
        plot_letter_basic(pts, label)

        # Experiment 1: node counts
        experiment_node_counts(pts, label)

        # Experiment 2: smoothing parameter
        experiment_smoothing(pts, label)


if __name__ == "__main__":
    main()
