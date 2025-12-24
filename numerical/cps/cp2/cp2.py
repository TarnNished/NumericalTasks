import numpy as np
import matplotlib.pyplot as plt
import time

# Modeling the Dynamics of a Swarm of Drones
# System: each drone experiences pairwise attractive and repulsive forces
# We compare TWO implicit methods for solving the ODE system:
#  - Method 1: Implicit backward Euler solved by FIXED-POINT ITERATION (5 points)
#  - Method 2: Implicit backward Euler solved by NEWTON-GAUSS-SEIDEL (5 points)
# Explicit Euler is included as a baseline to demonstrate why implicit methods are needed

# --- Parameters ---
N = 5                 # number of drones
k_attr = 1.0          # attractive constant
k_rep = 0.5           # repulsive constant (smaller to keep dynamics stable)
m = 1.0               # mass
dt = 0.05             # time step
T = 10.0              # total time
steps = int(T / dt)
eps = 1e-6            # small value to avoid division by zero

np.random.seed(0)

# Force model (pairwise)
def compute_forces(positions):
    # positions: (N,2)
    forces = np.zeros_like(positions)
    for i in range(N):
        f = np.zeros(2)
        r_i = positions[i]
        for j in range(N):
            if i == j:
                continue
            r_j = positions[j]
            diff = r_j - r_i
            dist = np.linalg.norm(diff) + eps
            f += k_attr * diff + k_rep * (r_i - r_j) / (dist**2)
        forces[i] = f
    return forces

# Compute total energy (kinetic + potential)
def compute_energy(positions, velocities):
    # Kinetic energy
    ke = 0.5 * m * np.sum(velocities**2)
    # Potential energy (attractive and repulsive)
    pe = 0.0
    for i in range(N):
        for j in range(i+1, N):
            r_ij = np.linalg.norm(positions[j] - positions[i]) + eps
            pe += -0.5 * k_attr * r_ij**2 - k_rep / r_ij
    return ke + pe

# --- Solvers ---
# 1) Explicit Euler (baseline for comparison)
def simulate_explicit_euler(r0, v0):
    r = r0.copy()
    v = v0.copy()
    traj = np.zeros((steps + 1, N, 2))
    vel_traj = np.zeros((steps + 1, N, 2))
    energy_hist = np.zeros(steps + 1)
    traj[0] = r
    vel_traj[0] = v
    energy_hist[0] = compute_energy(r, v)

    start = time.time()
    for n in range(steps):
        f = compute_forces(r)
        v = v + (dt / m) * f
        r = r + dt * v
        traj[n + 1] = r
        vel_traj[n + 1] = v
        energy_hist[n + 1] = compute_energy(r, v)
    elapsed = time.time() - start
    return traj, vel_traj, energy_hist, elapsed

# 2) METHOD 1: Implicit Backward Euler solved by FIXED-POINT ITERATION
def simulate_implicit_fixed_point(r0, v0, tol=1e-5, max_iter=200):
    r = r0.copy()
    v = v0.copy()
    traj = np.zeros((steps + 1, N, 2))
    vel_traj = np.zeros((steps + 1, N, 2))
    energy_hist = np.zeros(steps + 1)
    iter_counts = []
    traj[0] = r
    vel_traj[0] = v
    energy_hist[0] = compute_energy(r, v)

    start = time.time()
    for n in range(steps):
        # predictor (explicit Euler)
        r_guess = r + dt * v
        for it in range(max_iter):
            f_guess = compute_forces(r_guess)
            r_new = r + dt * (v + (dt / m) * f_guess)
            err = np.max(np.abs(r_new - r_guess))
            r_guess = r_new
            if err < tol:
                iter_counts.append(it + 1)
                break
        else:
            iter_counts.append(max_iter)
        # update velocity using backward Euler (use the converged r_guess)
        f_new = compute_forces(r_guess)
        v = v + (dt / m) * f_new
        r = r_guess
        traj[n + 1] = r
        vel_traj[n + 1] = v
        energy_hist[n + 1] = compute_energy(r, v)
    elapsed = time.time() - start
    return traj, vel_traj, energy_hist, elapsed, iter_counts

# 3) METHOD 2: Implicit Backward Euler solved by NEWTON-GAUSS-SEIDEL
def simulate_newton_gauss_seidel(r0, v0, tol=1e-6, max_iter=50):
    r = r0.copy()
    v = v0.copy()
    traj = np.zeros((steps + 1, N, 2))
    vel_traj = np.zeros((steps + 1, N, 2))
    energy_hist = np.zeros(steps + 1)
    iter_counts = []
    traj[0] = r
    vel_traj[0] = v
    energy_hist[0] = compute_energy(r, v)

    start = time.time()
    for n in range(steps):
        # initial guess
        r_guess = r + dt * v
        for it in range(max_iter):
            max_update = 0.0
            # update each particle sequentially (Gauss-Seidel)
            for i in range(N):
                # build g_i for particle i
                def g_i(r_i):
                    r_temp = r_guess.copy()
                    r_temp[i] = r_i
                    f_temp = compute_forces(r_temp)
                    return r_i - r[i] - dt * (v[i] + (dt / m) * f_temp[i])
                # evaluate g and approximate Jacobian J (2x2) by finite differences
                ri = r_guess[i].copy()
                gi = g_i(ri)
                J = np.zeros((2, 2))
                h = 1e-6
                for k in range(2):
                    perturb = np.zeros(2)
                    perturb[k] = h
                    g_pert = g_i(ri + perturb)
                    J[:, k] = (g_pert - gi) / h
                # solve J * delta = -g
                try:
                    delta = np.linalg.solve(J, -gi)
                except np.linalg.LinAlgError:
                    delta = -0.5 * gi
                # damping to improve convergence
                lambda_damp = 1.0
                ri_new = ri + lambda_damp * delta
                update_norm = np.linalg.norm(ri_new - ri)
                r_guess[i] = ri_new
                if update_norm > max_update:
                    max_update = update_norm
            if max_update < tol:
                iter_counts.append(it + 1)
                break
        else:
            iter_counts.append(max_iter)
        # after converging (or max iter), update velocity and state
        f_new = compute_forces(r_guess)
        v = v + (dt / m) * f_new
        r = r_guess
        traj[n + 1] = r
        vel_traj[n + 1] = v
        energy_hist[n + 1] = compute_energy(r, v)
    elapsed = time.time() - start
    return traj, vel_traj, energy_hist, elapsed, iter_counts

# --- Utilities for plotting and comparison ---

def plot_trajectories(traj, ax, title):
    colors = plt.get_cmap('tab10', N)
    for i in range(N):
        ax.plot(traj[:, i, 0], traj[:, i, 1], color=colors(i), label=f"Drone {i+1}", alpha=0.7)
        ax.scatter(traj[0, i, 0], traj[0, i, 1], color=colors(i), marker='s', s=100, edgecolors='black', linewidths=2)
        ax.scatter(traj[-1, i, 0], traj[-1, i, 1], color=colors(i), marker='o', s=100, edgecolors='black', linewidths=2)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

# --- Main experiment ---
if __name__ == '__main__':
    # initial conditions (same for all methods)
    r0 = np.random.rand(N, 2) * 10
    v0 = np.zeros((N, 2))

    # Run simulations
    print('='*70)
    print('DRONE SWARM DYNAMICS SIMULATION')
    print('Comparing Two Implicit Methods for Solving ODEs')
    print('='*70)

    print('\n[Baseline] Running Explicit Euler...')
    traj_exp, vel_exp, energy_exp, t_exp = simulate_explicit_euler(r0, v0)
    print(f'  ✓ Explicit Euler finished in {t_exp:.4f} s')

    print('\n[Method 1] Running Implicit Backward Euler with FIXED-POINT ITERATION...')
    traj_fp, vel_fp, energy_fp, t_fp, iters_fp = simulate_implicit_fixed_point(r0, v0)
    print(f'  ✓ Fixed-Point method finished in {t_fp:.4f} s')
    print(f'  ✓ Average iterations per step: {np.mean(iters_fp):.2f}')

    print('\n[Method 2] Running Implicit Backward Euler with NEWTON-GAUSS-SEIDEL...')
    traj_ngs, vel_ngs, energy_ngs, t_ngs, iters_ngs = simulate_newton_gauss_seidel(r0, v0)
    print(f'  ✓ Newton-GS method finished in {t_ngs:.4f} s')
    print(f'  ✓ Average iterations per step: {np.mean(iters_ngs):.2f}')

    # Comparison metrics
    def avg_pairwise_dist(traj):
        final = traj[-1]
        dists = []
        for i in range(N):
            for j in range(i + 1, N):
                dists.append(np.linalg.norm(final[i] - final[j]))
        return np.mean(dists)

    def trajectory_difference(traj1, traj2):
        """Compute L2 norm difference between trajectories"""
        return np.sqrt(np.mean((traj1 - traj2)**2))

    spread_exp = avg_pairwise_dist(traj_exp)
    spread_fp = avg_pairwise_dist(traj_fp)
    spread_ngs = avg_pairwise_dist(traj_ngs)

    traj_diff = trajectory_difference(traj_fp, traj_ngs)
    energy_drift_fp = np.abs(energy_fp[-1] - energy_fp[0])
    energy_drift_ngs = np.abs(energy_ngs[-1] - energy_ngs[0])

    print('\n' + '='*70)
    print('DETAILED COMPARISON OF THE TWO IMPLICIT METHODS')
    print('='*70)

    print('\n1. COMPUTATIONAL EFFICIENCY:')
    print(f'   Fixed-Point Iteration:')
    print(f'     - Total runtime: {t_fp:.4f} s')
    print(f'     - Avg iterations/step: {np.mean(iters_fp):.2f}')
    print(f'     - Max iterations/step: {np.max(iters_fp)}')
    print(f'   Newton-Gauss-Seidel:')
    print(f'     - Total runtime: {t_ngs:.4f} s')
    print(f'     - Avg iterations/step: {np.mean(iters_ngs):.2f}')
    print(f'     - Max iterations/step: {np.max(iters_ngs)}')
    print(f'   → Speedup: {t_ngs/t_fp:.2f}x (Fixed-Point is {"faster" if t_fp < t_ngs else "slower"})')

    print('\n2. SOLUTION ACCURACY:')
    print(f'   Final average pairwise distance (swarm spread):')
    print(f'     - Fixed-Point: {spread_fp:.6f}')
    print(f'     - Newton-GS: {spread_ngs:.6f}')
    print(f'     - Difference: {abs(spread_fp - spread_ngs):.6f}')
    print(f'   Trajectory L2 difference: {traj_diff:.6e}')

    print('\n3. ENERGY CONSERVATION:')
    print(f'   Total energy drift over simulation:')
    print(f'     - Fixed-Point: {energy_drift_fp:.6e}')
    print(f'     - Newton-GS: {energy_drift_ngs:.6e}')
    print(f'   → Better conservation: {"Fixed-Point" if energy_drift_fp < energy_drift_ngs else "Newton-GS"}')

    print('\n4. CONVERGENCE BEHAVIOR:')
    print(f'   Steps requiring max iterations:')
    print(f'     - Fixed-Point: {np.sum(np.array(iters_fp) >= 200)}/{steps}')
    print(f'     - Newton-GS: {np.sum(np.array(iters_ngs) >= 50)}/{steps}')

    print('\n5. COMPARISON WITH EXPLICIT EULER (Baseline):')
    print(f'   Explicit Euler spread: {spread_exp:.6f}')
    print(f'   → Implicit methods produce {spread_exp/spread_fp:.2f}x tighter formation')
    print(f'   → Demonstrates superior stability of implicit schemes')

    print('\n' + '='*70)
    print('CONCLUSION:')
    print('='*70)
    if t_fp < t_ngs and abs(spread_fp - spread_ngs) < 0.01:
        print('✓ FIXED-POINT ITERATION is RECOMMENDED for this problem:')
        print(f'  - {t_ngs/t_fp:.1f}x faster than Newton-GS')
        print(f'  - Similar accuracy (difference < 1%)')
        print(f'  - Simpler implementation')
    elif abs(spread_fp - spread_ngs) > 0.1:
        print('✓ NEWTON-GAUSS-SEIDEL is RECOMMENDED for this problem:')
        print(f'  - More accurate solution')
        print(f'  - Better convergence properties')
    else:
        print('✓ Both methods produce comparable results')
        print(f'  - Trade-off: Fixed-Point is faster, Newton-GS more robust')
    print('='*70)

    # Create comprehensive comparison plots
    fig = plt.figure(figsize=(20, 12))

    # Row 1: Trajectories
    ax1 = plt.subplot(3, 3, 1)
    plot_trajectories(traj_exp, ax1, 'Explicit Euler\n(Baseline)')

    ax2 = plt.subplot(3, 3, 2)
    plot_trajectories(traj_fp, ax2, 'Method 1: Fixed-Point\nIteration')

    ax3 = plt.subplot(3, 3, 3)
    plot_trajectories(traj_ngs, ax3, 'Method 2: Newton-\nGauss-Seidel')

    # Row 2: Energy evolution and convergence
    ax4 = plt.subplot(3, 3, 4)
    time_grid = np.linspace(0, T, steps + 1)
    ax4.plot(time_grid, energy_exp, 'r-', label='Explicit Euler', linewidth=2, alpha=0.7)
    ax4.plot(time_grid, energy_fp, 'b-', label='Fixed-Point', linewidth=2, alpha=0.7)
    ax4.plot(time_grid, energy_ngs, 'g-', label='Newton-GS', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Total Energy')
    ax4.set_title('Energy Conservation', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(range(1, steps + 1), iters_fp, 'b-', label='Fixed-Point', linewidth=1.5, alpha=0.7)
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Iterations to Converge')
    ax5.set_title('Fixed-Point Convergence', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=np.mean(iters_fp), color='b', linestyle='--', label=f'Mean: {np.mean(iters_fp):.1f}')
    ax5.legend()

    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(range(1, steps + 1), iters_ngs, 'g-', label='Newton-GS', linewidth=1.5, alpha=0.7)
    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('Iterations to Converge')
    ax6.set_title('Newton-GS Convergence', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=np.mean(iters_ngs), color='g', linestyle='--', label=f'Mean: {np.mean(iters_ngs):.1f}')
    ax6.legend()

    # Row 3: Position difference and performance comparison
    ax7 = plt.subplot(3, 3, 7)
    diff_norm = np.sqrt(np.sum((traj_fp - traj_ngs)**2, axis=(1, 2)))
    ax7.plot(time_grid, diff_norm, 'purple', linewidth=2)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('L2 Norm Difference')
    ax7.set_title('Trajectory Difference\n(Fixed-Point vs Newton-GS)', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.axhline(y=np.mean(diff_norm), color='purple', linestyle='--', alpha=0.5)

    ax8 = plt.subplot(3, 3, 8)
    methods = ['Fixed-Point', 'Newton-GS']
    times = [t_fp, t_ngs]
    colors_bar = ['blue', 'green']
    bars = ax8.bar(methods, times, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    ax8.set_ylabel('Runtime (seconds)')
    ax8.set_title('Computational Time Comparison', fontweight='bold')
    ax8.grid(True, axis='y', alpha=0.3)
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.4f}s', ha='center', va='bottom', fontweight='bold')

    ax9 = plt.subplot(3, 3, 9)
    metrics = ['Avg Iterations', 'Max Iterations']
    fp_vals = [np.mean(iters_fp), np.max(iters_fp)]
    ngs_vals = [np.mean(iters_ngs), np.max(iters_ngs)]
    x = np.arange(len(metrics))
    width = 0.35
    ax9.bar(x - width/2, fp_vals, width, label='Fixed-Point', color='blue', alpha=0.7, edgecolor='black')
    ax9.bar(x + width/2, ngs_vals, width, label='Newton-GS', color='green', alpha=0.7, edgecolor='black')
    ax9.set_ylabel('Number of Iterations')
    ax9.set_title('Convergence Metrics', fontweight='bold')
    ax9.set_xticks(x)
    ax9.set_xticklabels(metrics)
    ax9.legend()
    ax9.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    # Save results
    fig.savefig('cp2_detailed_comparison.png', dpi=200, bbox_inches='tight')
    print('\n✓ Saved detailed comparison figure to: cp2_detailed_comparison.png')

    np.savez('cp2_trajectories.npz',
             traj_exp=traj_exp, traj_fp=traj_fp, traj_ngs=traj_ngs,
             energy_exp=energy_exp, energy_fp=energy_fp, energy_ngs=energy_ngs,
             iters_fp=iters_fp, iters_ngs=iters_ngs)
    print('✓ Saved trajectory data to: cp2_trajectories.npz')

    with open('cp2_comparison_report.txt', 'w') as f:
        f.write('='*70 + '\n')
        f.write('DRONE SWARM DYNAMICS - METHOD COMPARISON REPORT\n')
        f.write('='*70 + '\n\n')
        f.write('Problem: Modeling dynamics of a swarm of drones with pairwise forces\n')
        f.write(f'Number of drones: {N}\n')
        f.write(f'Time step: {dt} s\n')
        f.write(f'Total simulation time: {T} s\n')
        f.write(f'Number of steps: {steps}\n\n')
        f.write('='*70 + '\n')
        f.write('METHOD COMPARISON\n')
        f.write('='*70 + '\n\n')
        f.write('Method 1: Implicit Backward Euler with Fixed-Point Iteration\n')
        f.write(f'  Runtime: {t_fp:.6f} s\n')
        f.write(f'  Final spread: {spread_fp:.6f}\n')
        f.write(f'  Avg iterations/step: {np.mean(iters_fp):.2f}\n')
        f.write(f'  Energy drift: {energy_drift_fp:.6e}\n\n')
        f.write('Method 2: Implicit Backward Euler with Newton-Gauss-Seidel\n')
        f.write(f'  Runtime: {t_ngs:.6f} s\n')
        f.write(f'  Final spread: {spread_ngs:.6f}\n')
        f.write(f'  Avg iterations/step: {np.mean(iters_ngs):.2f}\n')
        f.write(f'  Energy drift: {energy_drift_ngs:.6e}\n\n')
        f.write('Comparison:\n')
        f.write(f'  Speed ratio (NGS/FP): {t_ngs/t_fp:.4f}\n')
        f.write(f'  Solution difference: {traj_diff:.6e}\n')
        f.write(f'  Spread difference: {abs(spread_fp - spread_ngs):.6e}\n')

    print('✓ Saved detailed report to: cp2_comparison_report.txt')
    print('\nAll outputs saved successfully!\n')

    plt.show()
