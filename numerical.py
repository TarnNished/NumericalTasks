import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return np.exp(-x) * np.cos(2 * x)


def exact_derivative(x):
    return -np.exp(-x) * (np.cos(2 * x) + 2 * np.sin(2 * x))


def second_order_fd(x0, step_size):

    x1 = x0 + step_size
    x2 = x0 + 3 * step_size


    return (-func(x0) + 4 * func(x1) - 3 * func(x2)) / (2 * step_size)



points = np.array([0, 0.5, 1.0, 1.5, 2.0])
step_size = points[1] - points[0]  # Step size

exact_derivative_vals = exact_derivative(points)

second_order_fd_vals = np.array([second_order_fd(x, step_size) for x in points])

error_vector = second_order_fd_vals - exact_derivative_vals


l1_norm = np.sum(np.abs(error_vector))


l2_norm = np.sqrt(np.sum(error_vector ** 2))


infinity_norm = np.max(np.abs(error_vector))



print(f"Points: {points}")
print(f"Exact Derivative Values: {exact_derivative_vals}")
print(f"Approximated Derivative Values (2nd order FD): {second_order_fd_vals}")
print(f"Error Vector (2nd order FD): {error_vector}")
print(f"\nError Norms (2nd order FD):")
print(f"L1 norm: {l1_norm:.6f}")
print(f"L2 norm: {l2_norm:.6f}")
print(f"Infinity norm: {infinity_norm:.6f}")


plt.plot(points, exact_derivative_vals, label="Exact derivative", marker='o')
plt.plot(points, second_order_fd_vals, label="Approximated derivative (2nd order FD)", marker='x')
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.legend()
plt.title("Exact vs Approximated Derivative (2nd order FD)")
plt.grid(True)
plt.show()



def forward_fd(x0, step_size):

    x1 = x0 + step_size


    return (func(x1) - func(x0)) / step_size



forward_fd_vals = np.array([forward_fd(x, step_size) for x in points])


error_vector_forward = forward_fd_vals - exact_derivative_vals


l1_norm_forward = np.sum(np.abs(error_vector_forward))

l2_norm_forward = np.sqrt(np.sum(error_vector_forward ** 2))

infinity_norm_forward = np.max(np.abs(error_vector_forward))


print(f"Approximated Derivative Values (Forward FD): {forward_fd_vals}")
print(f"Error Vector (Forward FD): {error_vector_forward}")
print(f"\nError Norms (Forward FD):")
print(f"L1 norm (Forward FD): {l1_norm_forward:.6f}")
print(f"L2 norm (Forward FD): {l2_norm_forward:.6f}")
print(f"Infinity norm (Forward FD): {infinity_norm_forward:.6f}")

plt.plot(points, exact_derivative_vals, label="Exact derivative", marker='o')
plt.plot(points, second_order_fd_vals, label="Approximated derivative (2nd order FD)", marker='x')
plt.plot(points, forward_fd_vals, label="Forward difference derivative", marker='s')
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.legend()
plt.title("Exact vs Approximated Derivatives (2nd order FD and Forward FD)")
plt.grid(True)
plt.show()


F_exact = func(points)

F_approx_second_order = np.array([second_order_fd(x, step_size) for x in points])
F_forward = np.array([forward_fd(x, step_size) for x in points])

E_second_order = F_approx_second_order - exact_derivative_vals
E_forward = F_forward - exact_derivative_vals

frobenius_norm_second_order = np.sqrt(np.sum(E_second_order**2))
frobenius_norm_forward = np.sqrt(np.sum(E_forward**2))

norm_1_second_order = np.max(np.sum(np.abs(E_second_order), axis=0))
norm_1_forward = np.max(np.sum(np.abs(E_forward), axis=0))


norm_inf_second_order = np.max(np.abs(E_second_order))
norm_inf_forward = np.max(np.abs(E_forward))

print(f"Error Matrix (2nd order FD): {E_second_order}")
print(f"Error Matrix (Forward FD): {E_forward}")
print(f"\nFrobenius Norms:")
print(f"Frobenius norm (2nd order FD): {frobenius_norm_second_order:.6f}")
print(f"Frobenius norm (Forward FD): {frobenius_norm_forward:.6f}")

print(f"\n1-norms:")
print(f"1-norm (2nd order FD): {norm_1_second_order:.6f}")
print(f"1-norm (Forward FD): {norm_1_forward:.6f}")

print(f"\nInfinity Norms:")
print(f"Infinity norm (2nd order FD): {norm_inf_second_order:.6f}")
print(f"Infinity norm (Forward FD): {norm_inf_forward:.6f}")


plt.plot(points, np.abs(E_second_order), label="Absolute error (2nd order FD)", marker='o')
plt.plot(points, np.abs(E_forward), label="Absolute error (Forward FD)", marker='x')
plt.xlabel("x")
plt.ylabel("Absolute error")
plt.legend()
plt.title("Comparison of Absolute Errors for Both Methods")
plt.grid(True)
plt.show()


absolute_error_second_order = np.abs(E_second_order)
absolute_error_forward = np.abs(E_forward)

plt.plot(points, absolute_error_second_order, label="Absolute error (2nd order FD)", marker='o')
plt.plot(points, absolute_error_forward, label="Absolute error (Forward FD)", marker='x')
plt.xlabel("x")
plt.ylabel("Absolute error")
plt.legend()
plt.title("Comparison of Absolute Errors for Both Methods")
plt.grid(True)
plt.show()
