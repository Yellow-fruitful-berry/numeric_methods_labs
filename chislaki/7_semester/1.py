import numpy as np
from matplotlib import pyplot as plt


def f(x,y):
    return -1*y/(2*x) + x*x


def df(x):
    return (5 / np.sqrt(x) + 2 * x * x * x) / 7


figure, axis = plt.subplots(1, 2, figsize=(10, 8))

h = 0.1
arr = np.arange(1+h, 2+h, h, dtype=float)
arr_ = np.arange(1, 2+h, h, dtype=float)

y_arr = [1]


for i in range(len(arr)):
    y_arr.append(y_arr[i] + h * f(arr[i], y_arr[i]))

# print(arr_)
# print(y_arr)
axis[0].plot(arr_, y_arr, label='h=0.1')


h = 0.01
arr = np.arange(1+h, 2+h, h, dtype=float)
arr_ = np.arange(1, 2+h, h, dtype=float)

y_arr = [1]


for i in range(len(arr)):
    y_arr.append(y_arr[i] + h * f(arr[i], y_arr[i]))

# print(arr_)
# print(y_arr)
axis[0].plot(arr_, y_arr, label='h=0.01')


arr = np.arange(1, 2.001, 0.001, dtype=float)
y_arr = []
for i in range(len(arr)):
    y_arr.append(df(arr[i]))
axis[0].plot(arr, y_arr, label='real')
leg = plt.legend()
axis[0].set_title("Метод Эйлера")






h = 0.1
arr = np.arange(1+h, 2+h, h, dtype=float)
arr_ = np.arange(1, 2+h, h, dtype=float)
y_arr = [1]


for i in range(len(arr) - 1):
    y_arr.append(y_arr[i] + h * 0.5 * (f(arr_[i], y_arr[i]) + f(arr_[i + 1], y_arr[i] + h * f(arr_[i], y_arr[i]))))

# print(arr_)
# print(y_arr)
axis[1].plot(arr_[:len(arr_)-1], y_arr, label='h=0.1')


h = 0.01
arr = np.arange(1+h, 2+h, h, dtype=float)
arr_ = np.arange(1, 2+h, h, dtype=float)

y_arr = [1]


for i in range(len(arr) - 1):
    y_arr.append(y_arr[i] + h * 0.5 * (f(arr_[i], y_arr[i]) + f(arr_[i + 1], y_arr[i] + h * f(arr_[i], y_arr[i]))))

# print(arr_)
# print(y_arr)
axis[1].plot(arr_[:len(arr_)-1], y_arr, label='h=0.01')


arr = np.arange(1, 2.001, 0.001, dtype=float)
y_arr = []
for i in range(len(arr)):
    y_arr.append(df(arr[i]))
axis[1].plot(arr, y_arr, label='real')
leg = plt.legend()
axis[1].set_title("Метод Эйлера с пересчетом")
plt.show()
