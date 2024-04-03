import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Задаем параметры
a = 1
b = 1
mu = 1
Lx = np.pi  # Обрезаем границы по x
Ly = np.pi

# Число узлов по x, y и t
Nx = 50  # Изменяем количество узлов по x
Ny = 50
Nt = 10  # Уменьшаем количество моментов времени

# Шаги по x, y и t
dx = Lx / (Nx - 1)  # Пересчитываем шаг по x
dy = Ly / (Ny - 1)
dt = 1.0  # Делаем каждый момент времени фиксированным

# Инициализация сетки
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)

# Создание графиков
fig = plt.figure(figsize=(12, 8))

# Основной цикл по времени
for n in range(Nt):
    # Инициализация функции u на начальном временном слое
    u = np.zeros((Nx, Ny))
    u_new = np.zeros((Nx, Ny))

    # Установка начальных условий u(x, y, 0) = 0
    if n == 0:
        u[:, :] = 0
    else:
        # Применение метода переменных направлений
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                u_new[i, j] = u[i, j] + (dt / 2) * (
                        a * ((u[i - 1, j] - 2 * u[i, j] + u[i + 1, j]) / dx**2) +
                        b * ((u[i, j - 1] - 2 * u[i, j] + u[i, j + 1]) / dy**2) +
                        np.sin(x[i]) * np.sin(y[j]) * (mu * np.cos(mu * (n * dt)) + (a + b) * np.sin(mu * (n * dt)))
                )

        # Применение граничных условий u(0, y, t) = u(x, 0, t) = 0
        for i in range(Nx):
            u_new[i, 0] = 0
        for j in range(Ny):
            u_new[0, j] = 0

        # Обновление значений с использованием промежуточного шага
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                u[i, j] = u_new[i, j] + (dt / 2) * (
                        a * ((u_new[i - 1, j] - 2 * u_new[i, j] + u_new[i + 1, j]) / dx**2) +
                        b * ((u_new[i, j - 1] - 2 * u_new[i, j] + u_new[i, j + 1]) / dy**2) +
                        np.sin(x[i]) * np.sin(y[j]) * (mu * np.cos(mu * (n * dt)) + (a + b) * np.sin(mu * (n * dt)))
                )

    # Создание подграфика
    ax = fig.add_subplot(2, 5, n + 1, projection='3d')
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, u, cmap='viridis')
    ax.set_title(f'Time: {n * dt}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('u(x, y, t)')

    # Обрезаем границы графика по оси X
    ax.set_xlim([0, Lx])
    ax.set_ylim ([0, Ly ])

plt.tight_layout()
plt.show()
