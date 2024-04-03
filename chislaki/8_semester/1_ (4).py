import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Параметры задачи
    a = 2.0  # коэффициент a
    b = 1.0  # коэффициент b
    N = 100  # количество узлов в сетке
    T = 0.01  # время
    t = np.linspace(0, 1, N + 1)

    # Создаем сетку
    x = np.linspace(0, np.pi, N + 1)
    dx = x[1] - x[0]
    dt = T / N

    u = np.zeros((N + 1, N + 1))

    # Начальное и граничные условия
    u[:, 0] = np.cos(x)  # Граничное условие u(x, 0) = cos(x)
    u[0, :] = np.exp(-a * t) * (np.cos(b * t) + np.sin(b * t))
    u[-1, :] = np.exp(-a * t) * (np.cos(b * t) + np.sin(b * t))

    for n in range(N):
        for i in range(1, N):
            uxx = (u[i + 1, n] - 2 * u[i, n] + u[i - 1, n]) / dx ** 2
            ux = (u[i + 1, n] - u[i - 1, n]) / (2 * dx)
            ut = a * uxx + b * ux
            u[i, n + 1] = u[i, n] + dt * ut

    X, Y = np.meshgrid(x, t)
    u[:, 0] = np.cos(x)
    print(u[:, 1])
    plt.plot(np.linspace(0, np.pi, len(u[:, 0])), u[:, 0])
    Z = u

    # Создадим 3D график
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Y, X, Z, cmap='viridis')

    # Установим подписи осей
    ax.set_xlabel('X')
    ax.set_ylabel('T')
    ax.set_zlabel('U')

    # Покажем график
    plt.show()
