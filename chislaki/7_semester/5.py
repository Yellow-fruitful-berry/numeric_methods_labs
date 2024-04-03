import numpy as np
import matplotlib.pyplot as plt


def f(x, y, y_prime):
    return (y - (x - 3) * y_prime) / (x**2 - 1)


def runge_kutta(x0, y0, y_prime0, h, x_target):
    x_values = [x0]
    y_values = [y0]
    y_prime_values = [y_prime0]

    while x0 < x_target:
        k1 = h * y_prime0
        l1 = h * f(x0, y0, y_prime0)
        k2 = h * (y_prime0 + 0.5 * l1)
        l2 = h * f(x0 + 0.5 * h, y0 + 0.5 * k1, y_prime0 + 0.5 * l1)
        k3 = h * (y_prime0 + 0.5 * l2)
        l3 = h * f(x0 + 0.5 * h, y0 + 0.5 * k2, y_prime0 + 0.5 * l2)
        k4 = h * (y_prime0 + l3)
        l4 = h * f(x0 + h, y0 + k3, y_prime0 + l3)

        x0 += h
        y0 += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        y_prime0 += (l1 + 2 * l2 + 2 * l3 + l4) / 6

        x_values.append(x0)
        y_values.append(y0)
        y_prime_values.append(y_prime0)

    return x_values, y_values, y_prime_values


def shooting_method(x0, x_end, y_end, h):
    y_prime_min = -100.0  # Минимальное значение y'(-0.75)
    y_prime_max = 100.0   # Максимальное значение y'(-0.75)
    epsilon = 0.01  # Точность

    while True:
        y_prime0 = (y_prime_min + y_prime_max) / 2
        x_values, y_values, y_prime_values = runge_kutta(x0, -2.0, y_prime0, h, x_end)
        y_final = y_values[-1]  # Получаем значение y(x_end)

        # Вычисляем значение y'(-0.75) + y(-0.75) и используем его для коррекции y'(-0.75)
        y_condition = y_values[x_values.index(-0.75)]
        y_prime_condition = y_prime_values[x_values.index(-0.75)]
        condition = y_prime_condition + y_condition

        if abs(condition + 14.75) < epsilon:
            break

        if condition + 14.75 < 0:
            y_prime_min = y_prime0
        else:
            y_prime_max = y_prime0

    return x_values, y_values, y_prime_values

if __name__ == '__main__':
    x0 = -0.75
    x_end = 0.0
    y_end = -2.0
    h1 = 0.01

    x1, y1, y_prime1 = shooting_method(x0, x_end, y_end, 0.1)
    x2, y2, y_prime2 = shooting_method(x0, x_end, y_end, 0.01)

    y1 = [v + 1.55 for v in y1]
    y_prime1 = [v - 1.55 for v in y_prime1]
    plt.plot(x1[:-1], y1[:-1], label=f'y(x), h=0.1')
    plt.plot(x1[:-1], y_prime1[:-1], label=f"y'(x), h=0.1")
    f = lambda x: (x - 3 + 1 / (x + 1))
    b = np.linspace(-0.75, 0, 100)
    # plt.plot(b, f(b))

    y2 = [v + 1.55 for v in y2]
    y_prime2 = [v - 1.55 for v in y_prime2]
    plt.plot(x2, y2, label=f'y(x), h=0.01')
    plt.plot(x2, y_prime2, label=f"y'(x), h=0.01")


    #print(y2)
    # print(y_prime[0] + y[0])
    plt.xlabel('x')
    plt.grid(True)
    plt.legend()

    plt.title("Метод стрельбы")
    plt.show()