import numpy as np
from matplotlib import pyplot as plt


def f1(x, y, z):
    return z


def f2(x, y, z):
    return (-2 * x*x*x * z - y) / (x*x*x*x)


def real_f(x):
    return 2*np.sin((x-1)/x) + np.cos((x-1)/x)


def real_df(x):
    return (2*np.cos((-1 + x)/x) + np.sin((1 - x)/x))/(x*x)


def update_y(f1, f2, f3, f4, h):
    return h*(55*f1 - 59*f2 + 37*f3 - 9*f4) / 24


def runge(x, y, z, h):
    k1 = h * f1(x, y, z)
    l1 = h * f2(x, y, z)

    k2 = h * f1(x + h*0.5, y + 0.5 * k1, z + 0.5 * l1)
    l2 = h * f2(x + h*0.5, y + 0.5 * k1, z + 0.5 * l1)

    k3 = h * f1(x + h*0.5, y + 0.5 * k2, z + 0.5 * l2)
    l3 = h * f2(x + 0.5*h, y + 0.5 * k2, z + 0.5 * l2)

    k4 = h * f1(x + h, y + k3, z + l3)
    l4 = h * f2(x + h, y + k3, z + l3)

    K = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    L = (l1 + 2 * l2 + 2 * l3 + l4) / 6
    return K, L



h = 0.1
x_arr = np.arange(1, 2+h, h)
y_arr = [1.0]
z_arr = [2.0]

for i in range(1, len(x_arr)):
    K = runge(x_arr[i-1], y_arr[i-1], z_arr[i-1], h)[0]
    L = runge(x_arr[i-1], y_arr[i-1], z_arr[i-1], h)[1]

    y_arr.append(y_arr[i-1] + K)
    z_arr.append(z_arr[i-1] + L)


plt.plot(x_arr, y_arr, label='Numeric method, h=0.1')



h = 0.01
x_arr = np.arange(1, 2+h, h)
y_arr = [1.0]
z_arr = [2.0]

for i in range(1, len(x_arr)):
    K = runge(x_arr[i-1], y_arr[i-1], z_arr[i-1], h)[0]
    L = runge(x_arr[i-1], y_arr[i-1], z_arr[i-1], h)[1]

    y_arr.append(y_arr[i-1] + K)
    z_arr.append(z_arr[i-1] + L)


plt.plot(x_arr, y_arr, label='Numeric method, h=0.01')


y_real = [1.0]  # начальное условие
x_arr = np.arange(1, 2.001, 0.001)
for i in range(1, len(x_arr)):
    y_real.append(real_f(x_arr[i]))

plt.plot(x_arr, y_real, label='Real')

y_der = [real_df(1.0)]  # начальное условие
x_arr = np.arange(1, 2.001, 0.001)
for i in range(1, len(x_arr)):
    y_der.append(real_df(x_arr[i]))

plt.plot(x_arr, y_der, label='Derivative')


leg = plt.legend()
plt.show()
