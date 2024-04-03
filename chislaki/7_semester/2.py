import numpy as np
from matplotlib import pyplot as plt


def f(x, y):
    return (-1*y*x + x*(x*x + 1))/(x*x+1)


def real_f(x):
    return (1 + x * x + 2 / np.sqrt(1 + x * x)) / 3


def df(x):
    return (5 / np.sqrt(x) + 2 * x * x * x) / 7


def delta_y(x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + h*0.5, y + 0.5 * h * f(x, y))
    k3 = h * f(x + 0.5*h, y + 0.5 * k2)
    k4 = h * f(x + h, y + k3)

    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


h = 0.1
arr = np.arange(0, 1+h, h)
y_arr = [1]
y_real = [1]

for i in range(len (arr) - 1):
    y_arr.append(y_arr[i] + delta_y(arr[i], y_arr[i], h))
plt.plot(arr, y_arr, label='Метод Рунге-Кутты h=0.1')


h = 0.01
arr = np.arange(0, 1+h, h)
y_arr = [1]
y_real = [1]

for i in range(len (arr) - 1):
    y_arr.append(y_arr[i] + delta_y(arr[i], y_arr[i], h))
plt.plot(arr, y_arr, label='Метод Рунге-Кутты h=0.01')


arr = np.arange(0, 1.001, 0.001)
for i in range (len (arr) - 1):
    y_real.append(real_f(arr[i]))

plt.plot(arr, y_real, label='Real')
leg = plt.legend()
plt.show()
