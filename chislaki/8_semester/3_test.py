import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Задаем начальные данные
Lx = np.pi / 2
Ly = np.pi / 2
Nx = 50
Ny = 50
dx = Lx / Nx
dy = Ly / Ny
x = np.linspace(0, Lx, Nx+1)
y = np.linspace(0, Ly, Ny+1)

# Создаем сетку
X, Y = np.meshgrid(x, y)

# Инициализируем начальное условие
u = np.zeros((Ny+1, Nx+1))
u[:, 0] = np.exp(-y) * np.cos(y)
u[0, :] = np.cos(x)
u[-1, :] = 0
u[:, -1] = 0

# Выполняем итерации
max_iter = 1000
tolerance = 1e-5
for _ in range(max_iter):
    u_old = u.copy()
    for i in range(1, Ny):
        for j in range(1, Nx):
            u[i, j] = 1 / (2/(dx*dx) + 2/(dy*dy) + 3) * (
                (u_old[i+1, j] + u_old[i-1, j]) / (dy*dy) +
                (u_old[i, j+1] + u_old[i, j-1]) / (dx*dx) - 2 * (u_old[i, j+1] - u_old[i, j-1]) / (2*dx) - 3 * u_old[i, j])
    if np.max(np.abs(u - u_old)) < tolerance:
        break

# Построение 3D графика
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, u, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u(x, y)')
plt.title('')
plt.show()
