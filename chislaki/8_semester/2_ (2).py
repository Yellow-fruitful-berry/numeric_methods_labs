import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Заданные параметры
L = np.pi
T = 1
Nx = 50  # количество шагов по x
Nt = 50  # количество шагов по t
dx = L / Nx
dt = T / Nt

# Создаем сетку
x_values = np.linspace(0, L, Nx+1)
t_values = np.linspace(0, T, Nt+1)

# Инициализация сетки для u и u_t
u = np.zeros((Nx+1, Nt+1))
u_t = np.zeros((Nx+1, Nt+1))

# Начальные условия
u[:, 0] = np.cos(x_values)
u_t[:, 0] = -np.cos(x_values)

# Начально-краевые условия
u[0, :] = np.exp(-t_values)
u[-1, :] = -np.exp(-t_values)

# Параметр для ускорения расчета
gamma = dt / dx

# Решение методом конечных разностей
for n in range(0, Nt):
    for i in range(1, Nx):
        u_xx = (u[i+1, n] - 2*u[i, n] + u[i-1, n]) / (dx**2)
        u_x = (u[i+1, n] - u[i-1, n]) / (2*dx)
        u_t[i, n+1] = u_t[i, n] + dt * (u_xx + u_x - u[i, n] + np.sin(x_values[i]) * np.exp(-t_values[n]))

    u[1:-1, n+1] = u[1:-1, n] + dt * u_t[1:-1, n+1]

# Создание 3D графика
X, T = np.meshgrid(x_values, t_values)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, u.T, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
ax.set_title('3D График решения')
plt.show()
