import numpy as np
import matplotlib.pyplot as plt

#Defining Norm Functions
def vector_norm(x, p=2):
    return np.linalg.norm(x, ord=p)

def matrix_norm(A, p=2):
    return np.linalg.norm(A, ord=p)

#generate random data
np.random.seed(42)

x1 = np.random.rand(4)
x2 = np.random.rand(4)

A1 = x1.reshape(2, 2)
A2 = x2.reshape(2, 2)

print("x1 =", x1)
print("x2 =", x2)
print("\nA1 =\n", A1)
print("A2 =\n", A2)

# Computing Norm Distances
d_L1 = vector_norm(x1 - x2, p=1)
d_L2 = vector_norm(x1 - x2, p=2)
d_matrix_L1 = matrix_norm(A1 - A2, p=1)
d_matrix_L2 = matrix_norm(A1 - A2, p=2)

print("\n--- Distances ---")
print(f"Vector distance (L1): {d_L1:.4f}")
print(f"Vector distance (L2): {d_L2:.4f}")
print(f"Matrix distance (L1 induced): {d_matrix_L1:.4f}")
print(f"Matrix distance (L2 induced): {d_matrix_L2:.4f}")

# Visualizing Unit Balls
theta = np.linspace(0, 2 * np.pi, 400)
circle_x = np.cos(theta)
circle_y = np.sin(theta)

l1_x = np.linspace(-1, 1, 400)
l1_y1 = 1 - np.abs(l1_x)
l1_y2 = -1 + np.abs(l1_x)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(circle_x, circle_y)
plt.title("L2 (Euclidean) Unit Ball")
plt.axis("equal")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(l1_x, l1_y1, 'r')
plt.plot(l1_x, l1_y2, 'r')
plt.title("L1 (Manhattan) Unit Ball")
plt.axis("equal")
plt.grid(True)

plt.suptitle("Sections of Unit Balls for L1 and L2 Norms", fontsize=13)
plt.show()
