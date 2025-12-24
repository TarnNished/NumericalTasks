# Modeling the Dynamics of a Swarm of Drones Using Implicit Numerical Methods

**Course Project 2 - Numerical Methods**  
**Date:** December 24, 2025  
**Problem:** Modeling with Ordinary Differential Equations

---

## 1. Introduction

This project investigates the dynamics of a swarm of autonomous drones interacting through pairwise attractive and repulsive forces. The goal is to compare two implicit numerical methods for solving the resulting system of ordinary differential equations (ODEs). Implicit methods are essential for this problem due to the stiff nature of the repulsive forces, which can cause explicit methods to become unstable at reasonable time steps.

### 1.1 Problem Statement

We model **N = 5 drones** moving in 2D space, where each drone experiences:
- **Attractive forces** pulling drones toward each other (cohesion)
- **Repulsive forces** preventing collisions (separation)

The system is described by the following ODEs for each drone *i*:

```
dₜr_i = v_i
dₜv_i = (1/m) · F_i(r)
```

where the total force on drone *i* is:

```
F_i(r) = Σ[j≠i] [k_attr · (r_j - r_i) + k_rep · (r_i - r_j) / |r_j - r_i|²]
```

**Parameters:**
- N = 5 drones
- k_attr = 1.0 (attractive constant)
- k_rep = 0.5 (repulsive constant)
- m = 1.0 (mass)
- dt = 0.05 s (time step)
- T = 10.0 s (simulation time)

---

## 2. Mathematical Model

### 2.1 Force Model

The force model combines two physical phenomena:

1. **Attractive Force:** Linear spring-like attraction encouraging swarm cohesion
   ```
   F_attr(r_i, r_j) = k_attr · (r_j - r_i)
   ```

2. **Repulsive Force:** Inverse-square law preventing collision
   ```
   F_rep(r_i, r_j) = k_rep · (r_i - r_j) / |r_j - r_i|²
   ```

The repulsive term creates **stiffness** in the system, requiring implicit methods for numerical stability.

### 2.2 Energy Conservation

The total energy consists of kinetic and potential components:

```
E_total = KE + PE
KE = (1/2) · m · Σ|v_i|²
PE = Σ[i<j] [-0.5 · k_attr · |r_j - r_i|² - k_rep / |r_j - r_i|]
```

---

## 3. Numerical Methods

We compare **two implicit methods** for solving the backward Euler scheme at each time step.

### 3.1 Method 1: Fixed-Point Iteration (5 points)

**Algorithm:**
```
For each time step n:
  1. Initialize: r_guess = r_n + dt · v_n (predictor)
  2. Iterate until convergence:
     F_guess = compute_forces(r_guess)
     r_new = r_n + dt · (v_n + dt/m · F_guess)
     if |r_new - r_guess| < tol: break
     r_guess = r_new
  3. Update: v_{n+1} = v_n + dt/m · F(r_{n+1})
             r_{n+1} = r_guess
```

**Advantages:**
- Simple implementation
- No derivative calculations required
- Globally consistent updates

**Convergence:** Tolerance = 10⁻⁵, Max iterations = 200

### 3.2 Method 2: Newton-Gauss-Seidel (5 points)

**Algorithm:**
```
For each time step n:
  1. Initialize: r_guess = r_n + dt · v_n
  2. Iterate until convergence:
     For each particle i (Gauss-Seidel sweep):
       Solve: g_i(r_i) = 0 using Newton's method
       where g_i(r_i) = r_i - r_n,i - dt·(v_n,i + dt/m · F_i(r))
       Approximate Jacobian J_i by finite differences
       Update: r_i ← r_i - J_i⁻¹ · g_i(r_i)
     if max_update < tol: break
  3. Update velocity and position as in Method 1
```

**Advantages:**
- Particle-wise Newton updates (local nonlinear solve)
- Typically fewer iterations per step
- Better convergence for highly nonlinear problems

**Convergence:** Tolerance = 10⁻⁶, Max iterations = 50

### 3.3 Baseline: Explicit Euler

For comparison, we include explicit Euler to demonstrate the **instability** of explicit methods for this stiff problem:
```
v_{n+1} = v_n + dt/m · F(r_n)
r_{n+1} = r_n + dt · v_{n+1}
```

---

## 4. Experimental Results

### 4.1 Computational Efficiency

| Method | Runtime (s) | Avg Iter/Step | Max Iter/Step |
|--------|------------|---------------|---------------|
| **Fixed-Point** | 0.0907 | 3.46 | 13 |
| **Newton-GS** | 1.0837 | 3.29 | 7 |
| **Speedup** | **11.95×** | - | - |

**Result:** Fixed-Point is **11.95× faster** than Newton-GS for this problem.

### 4.2 Solution Accuracy

| Metric | Fixed-Point | Newton-GS | Difference |
|--------|------------|-----------|------------|
| **Final Spread** | 0.8454 | 0.8404 | 0.0050 (0.6%) |
| **Trajectory L2 Difference** | - | - | 1.175 × 10⁻³ |

**Result:** Both methods produce **nearly identical solutions** (< 1% difference).

### 4.3 Energy Conservation

| Method | Energy Drift |
|--------|--------------|
| **Fixed-Point** | 78.95 |
| **Newton-GS** | 78.92 |

**Result:** Newton-GS shows **marginally better** energy conservation (0.04% improvement).

### 4.4 Convergence Behavior

- **Fixed-Point:** 0/200 steps required max iterations (all converged)
- **Newton-GS:** 0/200 steps required max iterations (all converged)
- Both methods show **robust convergence** for this problem

### 4.5 Comparison with Explicit Euler

| Method | Final Spread | Stability |
|--------|--------------|-----------|
| **Explicit Euler** | 3.1546 | Unstable |
| **Implicit Methods** | ~0.84 | Stable |
| **Improvement** | **3.73×** tighter formation | ✓ |

**Result:** Implicit methods produce **3.73× tighter swarm formation**, demonstrating their **superior stability** for stiff ODEs.

---

## 5. Visualization and Analysis

The simulation produces a comprehensive 9-panel visualization showing:

**Row 1 - Trajectories:**
- Explicit Euler (baseline showing divergence)
- Fixed-Point (stable, compact swarm)
- Newton-GS (stable, nearly identical to Fixed-Point)

**Row 2 - Energy and Convergence:**
- Energy conservation over time for all three methods
- Fixed-Point iteration count per time step
- Newton-GS iteration count per time step

**Row 3 - Method Comparison:**
- Trajectory difference between Fixed-Point and Newton-GS
- Runtime comparison bar chart
- Convergence metrics comparison

**Key Observations:**
1. Explicit Euler trajectories diverge, showing instability
2. Both implicit methods maintain compact swarm formation
3. Fixed-Point requires slightly more iterations but is much faster overall
4. Energy drift is similar for both implicit methods

---

## 6. Conclusions

### 6.1 Method Comparison Summary

**Fixed-Point Iteration:**
- ✓ **11.9× faster** than Newton-GS
- ✓ **Similar accuracy** (< 1% difference)
- ✓ **Simpler implementation** (no Jacobian required)
- ✓ **Robust convergence** (~3-4 iterations per step)

**Newton-Gauss-Seidel:**
- ✓ Slightly **better energy conservation**
- ✓ Fewer iterations per step (3.29 vs 3.46)
- ✗ **10× slower** due to per-particle Jacobian evaluations
- ✓ More robust for highly nonlinear/stiff problems

### 6.2 Recommendation

**For this drone swarm problem, Fixed-Point Iteration is the superior choice** due to:
1. Significantly faster runtime (11.9× speedup)
2. Nearly identical accuracy to Newton-GS
3. Simpler implementation and maintenance
4. Adequate convergence properties for this force model

### 6.3 When to Use Newton-Gauss-Seidel

Newton-GS should be preferred when:
- The problem is highly stiff and fixed-point fails to converge
- Maximum accuracy is critical (e.g., long-term simulations)
- Computational time is not a constraint
- The system has strong coupling requiring local Newton updates

### 6.4 Importance of Implicit Methods

The comparison with explicit Euler demonstrates that **implicit methods are essential** for stiff ODE systems like drone swarm dynamics. The repulsive 1/r² forces create rapid variations that require:
- Unconditional stability (large time steps without divergence)
- Accurate resolution of equilibrium states
- Physical energy conservation


---



## Appendix: Code Repository

**Files Generated:**
- `cp2.py` - Main simulation code (315 lines)
- `cp2_detailed_comparison.png` - 9-panel visualization
- `cp2_trajectories.npz` - Numerical data archive
- `cp2_comparison_report.txt` - Detailed metrics summary

**How to Run:**
```bash
python cp2.py
```

**Output:** Console comparison metrics, visualization window, and saved files.

---

**End of Report**

