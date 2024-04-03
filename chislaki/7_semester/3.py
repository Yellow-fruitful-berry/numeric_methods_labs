import numpy as np
from matplotlib import pyplot as plt


def f(x, y):
    return (y*y+x*y)/(x*x)


def real_f(x):
    return x / (1-np.log(x))


def update_y(f1, f2, f3, f4, h):
    return h*(55*f1 - 59*f2 + 37*f3 - 9*f4) / 24


def runge(x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + h*0.5, y + 0.5 * k1)
    k3 = h * f(x + 0.5*h, y + 0.5 * k2)
    k4 = h * f(x + h, y + k3)

    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


h = 0.1
x_arr = np.arange(1, 2+h, h)
y_arr = [1.0]


for i in range(1, 4):
    y_arr.append(y_arr[i-1] + runge(x_arr[i-1], y_arr[i-1], h))


for i in range(4, len(x_arr)):
    y_arr.append(y_arr[i-1] +
                 update_y(
                          f(x_arr[i-1], y_arr[i-1]),
                          f(x_arr[i-2], y_arr[i-2]),
                          f(x_arr[i-3], y_arr[i-3]),
                          f(x_arr[i-4], y_arr[i-4]),
                          h)
                 )
plt.plot(x_arr, y_arr, label='Adams method')


y_real = [1.0]
x_arr = np.arange(1, 2.001, 0.001)
for i in range(len(x_arr) - 1):
    y_real.append(real_f(x_arr[i]))

plt.plot(x_arr, y_real, label='Real')
leg = plt.legend()
plt.show()
