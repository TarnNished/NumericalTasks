import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


x = sp.Symbol('x')
f = sp.sin(x) + x**2
f_prime_exact = sp.diff(f, x)

x0 = 1.0

f_val = float(f.subs(x, x0))
f_prime_val = float(f_prime_exact.subs(x, x0))

def finite_diff_1d(func, x0, h):
    return (func(x0 + h) - func(x0 - h)) / (2 * h)

hs = [1e-1, 1e-2, 1e-3, 1e-4]
errors_1d = []

for h in hs:
    approx = finite_diff_1d(lambda t: np.sin(t) + t**2, x0, h)
    error = abs(approx - f_prime_val)
    errors_1d.append(error)
    print(f"h={h:.0e} | Approx={approx:.6f} | Exact={f_prime_val:.6f} | Error={error:.2e}")

plt.figure()
plt.loglog(hs, errors_1d, marker='o')
plt.xlabel("Step size h")
plt.ylabel("Error")
plt.title("Finite Difference Accuracy for f(x)=sin(x)+x^2")
plt.grid(True)
plt.show()


x_vals = np.linspace(-1, 3, 100)
f_vals = np.sin(x_vals) + x_vals**2
tangent = f_val + f_prime_val * (x_vals - x0)

plt.figure()
plt.plot(x_vals, f_vals, label="f(x)")
plt.plot(x_vals, tangent, '--', label="Tangent line")
plt.scatter([x0], [f_val], color='red')
plt.legend()
plt.title("Tangent Line to f(x)")
plt.show()


x, y = sp.symbols('x y')
g = x**2 + y**2 - x*y
gx_exact = sp.diff(g, x)
gy_exact = sp.diff(g, y)

x0, y0 = 1.0, 2.0
g_val = float(g.subs({x: x0, y: y0}))
gx_val = float(gx_exact.subs({x: x0, y: y0}))
gy_val = float(gy_exact.subs({x: x0, y: y0}))

def finite_diff_2d(func, x0, y0, h):
    gx = (func(x0 + h, y0) - func(x0 - h, y0)) / (2 * h)
    gy = (func(x0, y0 + h) - func(x0, y0 - h)) / (2 * h)
    return gx, gy

for h in hs:
    gx_approx, gy_approx = finite_diff_2d(lambda a,b: a**2 + b**2 - a*b, x0, y0, h)
    ex, ey = abs(gx_approx - gx_val), abs(gy_approx - gy_val)
    print(f"h={h:.0e} | gx_err={ex:.2e}, gy_err={ey:.2e}")

X, Y = np.meshgrid(np.linspace(0, 3, 30), np.linspace(0, 3, 30))
Z = X**2 + Y**2 - X*Y
plane = g_val + gx_val*(X - x0) + gy_val*(Y - y0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis', label="Surface")
ax.plot_surface(X, Y, plane, alpha=0.5, color='red')
ax.scatter(x0, y0, g_val, color='black', s=50)
ax.set_title("Tangent Plane to g(x,y)")
plt.show()
