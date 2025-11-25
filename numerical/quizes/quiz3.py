import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x) = tan^2(x)
def f(x):
    return np.tan(x)**2

# Compute Chebyshev nodes on [-0.8, 0.8]
n = 3
j = np.arange(1, n + 1)
t_chebyshev = np.cos((2 * j - 1) * np.pi / (2 * n))  # Nodes on [-1, 1]
x_chebyshev = -0.8 + (t_chebyshev + 1) * (0.8 - (-0.8)) / 2  # Rescale to [-0.8, 0.8]

# Function values at the Chebyshev nodes
f_values = f(x_chebyshev)

# Construct the divided-difference table
def divided_difference(x_vals, f_vals):
    n = len(x_vals)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = f_vals
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i, j] = (diff_table[i + 1, j - 1] - diff_table[i, j - 1]) / (x_vals[i + j] - x_vals[i])
    return diff_table

# Get the divided difference table
dd_table = divided_difference(x_chebyshev, f_values)

# Newton interpolating polynomial in Newton form
def newton_polynomial(x, x_vals, dd_table):
    n = len(x_vals)
    result = dd_table[0, 0]
    term = 1
    for i in range(1, n):
        term *= (x - x_vals[i - 1])
        result += dd_table[0, i] * term
    return result

# Evaluate P2(-0.3)
x_to_evaluate = -0.3
P2_at_neg_03 = newton_polynomial(x_to_evaluate, x_chebyshev, dd_table)

# Plotting
x_range = np.linspace(-0.8, 0.8, 400)
y_exact = f(x_range)
y_newton = newton_polynomial(x_range, x_chebyshev, dd_table)

plt.figure(figsize=(10, 6))
plt.plot(x_range, y_exact, label=r'Original function $f(x) = \tan^2(x)$', color='blue')
plt.plot(x_range, y_newton, label=r'Newton Interpolating Polynomial $P_2(x)$', color='green')
plt.scatter(x_chebyshev, f_values, color='red', zorder=5, label="Chebyshev Nodes")

plt.title("Chebyshev Interpolation of $f(x) = \tan^2(x)$")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="best")
plt.grid(True)
plt.show()

print(f"Newton interpolating polynomial evaluated at x = -0.3: {P2_at_neg_03:.4f}")
