import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initial data
a = 1
eps = 0.001
L = 40
M = 40
T = 10
l1 = math.pi / 2
l2 = np.log(2)
l3 = 1
h1 = l1 / L
h2 = l2 / M
tau = l3 / T
u_mesh = np.zeros((L + 1, M + 1, T + 1))  # Finite-difference mesh


def tridiagonal_matrix_algorithm(a, b, c, d, n):
    p = np.zeros([n])
    q = np.zeros([n])
    x = np.zeros([n])
    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]
    for i in range(1, n):
        p[i] = -c[i] / (b[i] + a[i] * p[i - 1])
        q[i] = (d[i] - a[i] * q[i - 1]) / (b[i] + a[i] * p[i - 1])

    x[n - 1] = q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]
    return x


def create_u(u):
    # Boundary conditions
    for j in range(M + 1):
        for k in range(T + 1):
            u[0][j][k] = math.cosh(j * h2) * math.exp(-3 * a * tau * k)

    for i in range(L + 1):
        for k in range(T + 1):
            u[i][0][k] = math.cos(2 * i * h1) * math.exp(-3 * a * tau * k)

    for i in range(L + 1):  # Initial condition
        for j in range(M + 1):
            u[i][j][0] = math.cos(2 * i * h1) * math.cosh(j * h2)

    for i in range(L + 1):
        for k in range(T + 1):
            u[i][M][k] = (3 / 4) * math.cos(2 * i * h1) * math.exp(-3 * a * tau * k)

    for j in range(M + 1):
        for k in range(T + 1):
            u[L][j][k] = 0

    # Computation
    for k in range(1, T + 1):
        # Calculate (n + 1/2) layer
        for j in range(M + 1):
            A1 = []
            B1 = []
            C1 = []
            F1 = []
            for i in range(L + 1):
                A1.append((a * tau) / (2 * h1 ** 2))
                B1.append(-(1 + ((a * tau) / h1 ** 2)))
                C1.append((a * tau) / (2 * h1 ** 2))
                F1.append(-u[i][j][k - 1])
            mass1 = tridiagonal_matrix_algorithm(A1, B1, C1, F1, L + 1)
            for i in range(L + 1):
                u[i][j][k] = mass1[i]

        # Calculate (n + 1) layer
        for i in range(L + 1):
            A2 = []
            B2 = []
            C2 = []
            F2 = []
            for j in range(M + 1):
                A2.append((a * tau) / (2 * h2 ** 2))
                B2.append(-(1 + ((a * tau) / h2 ** 2)))
                C2.append((a * tau) / (2 * h2 ** 2))
                F2.append(-u[i][j][k])
            mass2 = tridiagonal_matrix_algorithm(A2, B2, C2, F2, M + 1)
            for j in range(M + 1):
                u[i][j][k] = mass2[j]

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(0, l1, L + 1)
    y = np.linspace(0, l2, M + 1)
    X, Y = np.meshgrid(x, y)

    for t in range(T + 1):
        Z = u[:, :, t]
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('u')
    ax.set_title('Solution for different time steps')

    plt.show()


create_u(u_mesh)
