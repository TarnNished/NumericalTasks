import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def f(x):
    return 1 / (x + 1)

x_nodes2 = np.array([0.0, 1.5, 3.0])
y_nodes2 = f(x_nodes2)

x_nodes3 = np.array([0.0, 1.5, 2.0, 3.0])
y_nodes3 = f(x_nodes3)


def lagrange_interpolation(x, x_nodes, y_nodes):
    x = np.array(x, dtype=float)
    total = np.zeros_like(x, dtype=float)

    n = len(x_nodes)
    for i in range(n):
        xi = x_nodes[i]
        yi = y_nodes[i]
        Li = np.ones_like(x, dtype=float)
        for j in range(n):
            if j != i:
                Li *= (x - x_nodes[j]) / (xi - x_nodes[j])
        total += yi * Li
    return total


def P2(x):
    return lagrange_interpolation(x, x_nodes2, y_nodes2)

def P3(x):
    return lagrange_interpolation(x, x_nodes3, y_nodes3)


x_test = np.array([0.75, 1.5, 2.25])



x0, x1, x2 = x_nodes2
f_x0, f_x1, f_x2 = y_nodes2

print("==== Step 1 ====")
print("Function: f(x) = 1/(x+1)")
print(f"x0 = {x0}, f(x0) = {f_x0}")
print(f"x1 = {x1}, f(x1) = {f_x1}")
print(f"x2 = {x2}, f(x2) = {f_x2}")
print()



A = np.vstack([x_nodes2**2, x_nodes2, np.ones_like(x_nodes2)]).T
coeffs2 = np.linalg.solve(A, y_nodes2)
a2, b2, c2 = coeffs2

print("==== Step 2 ====")
print("Quadratic interpolant P2(x) through (0,1), (1.5,0.4), (3,0.25)")
print("Lagrange form: sum f(xi)*Li(x)")
print(f"Simplified polynomial form:")
print(f"P2(x) = {a2} * x^2 + {b2} * x + {c2}")
print()


f_test = f(x_test)
P2_test = P2(x_test)
errors2 = np.abs(f_test - P2_test)

print("==== Step 3 ====")
for xi, fi, pi, ei in zip(x_test, f_test, P2_test, errors2):
    print(f"x = {xi}:  f(x) = {fi},  P2(x) = {pi},  error = {ei}")
print()


L2_norm = np.sqrt(np.sum(errors2**2))
Linf_norm = np.max(errors2)

print("==== Step 4 ====")
print(f"‖e‖₂ (discrete L2 norm over test points) = {L2_norm}")
print(f"‖e‖∞ (discrete L∞ norm over test points) = {Linf_norm}")
print()



B = np.vstack([x_nodes3**3, x_nodes3**2, x_nodes3, np.ones_like(x_nodes3)]).T
coeffs3 = np.linalg.solve(B, y_nodes3)
A3, B3, C3, D3 = coeffs3

print("==== Step 5 ====")
print("Added node x3 = 2 with f(x3) = 1/3.")
print("Cubic interpolant P3(x) through (0,1), (1.5,0.4), (2,1/3), (3,0.25).")
print("P3(x) = A x^3 + B x^2 + C x + D with:")
print(f"A = {A3}, B = {B3}, C = {C3}, D = {D3}")
print()


x_vals = np.linspace(0, 3, 300)
f_vals = f(x_vals)
P2_vals = P2(x_vals)
P3_vals = P3(x_vals)

plt.figure(figsize=(8,5))
plt.plot(x_vals, f_vals, label='f(x) = 1/(x+1)')
plt.plot(x_vals, P2_vals, '--', label='P₂(x) quadratic')
plt.plot(x_vals, P3_vals, '-.', label='P₃(x) cubic')
plt.scatter(x_nodes2, y_nodes2, color='red', zorder=5, label='P₂ nodes')
plt.scatter(x_nodes3, y_nodes3, color='black', marker='x', zorder=5, label='P₃ nodes')
plt.title("Step 6: Interpolation vs true function on [0,3]")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

err2_curve = np.abs(f_vals - P2_vals)
err3_curve = np.abs(f_vals - P3_vals)

plt.figure(figsize=(8,5))
plt.plot(x_vals, err2_curve, '--', label='|f - P₂|')
plt.plot(x_vals, err3_curve, '-.', label='|f - P₃|')
plt.title("Step 7: Pointwise interpolation error")
plt.xlabel("x")
plt.ylabel("absolute error")
plt.legend()
plt.grid(True)
plt.show()



P3_test = P3(x_test)
errors3 = np.abs(f_test - P3_test)

print("==== Step 8 ====")

print("Shape change after adding new node at x=2:")
print("- P₂(x) is a quadratic (parabola).")
print("- P₃(x) is a cubic, so it can 'bend' more and pass exactly through the extra node (2, 1/3).")
print("- Visually, P₃ tracks f(x) more closely in the middle of [0,3].")

print("\nInterpolation errors at test points:")
for xi, e2i, e3i in zip(x_test, errors2, errors3):
    better = "P₃" if e3i < e2i else "P₂"
    print(f"x = {xi}: |f-P2| = {e2i}, |f-P3| = {e3i} -> {better} is closer")


max_err2 = np.max(err2_curve)
max_err3 = np.max(err3_curve)
print("\nError distribution over [0,3]:")
print(f"max |f-P2| over [0,3] ≈ {max_err2}")
print(f"max |f-P3| over [0,3] ≈ {max_err3}")
if max_err3 < max_err2:
    print("- P₃ generally has lower maximum error on [0,3].")
else:
    print("- P₂ actually peaks lower in max error somewhere (check plots).")

