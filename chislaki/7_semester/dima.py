import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def f1(x, z):
    return x / np.sqrt(1.0 + np.power(x, 2) + np.power(z, 2))


def f2(x, y):
    return y / np.sqrt(1.0 + np.power(x, 2) + np.power(y, 2))


def runge_kutta(x0, y, x, z, h):
    n = int((x - x0) / h)
    Y = [y]*(n+1)
    Z = [z]*(n+1)
    one_six = 1.0 / 6.0
    for i in range(1, n + 1):
        k1 = h * f1(x0, z)
        l1 = h * f2(x0, y)
        k2 = h * f1(x0 + 0.5 * h, z + 0.5 * k1)
        l2 = h * f2(x0 + 0.5 * h, y + 0.5 * l1)
        k3 = h * f1(x0 + 0.5 * h, z + 0.5 * k2)
        l3 = h * f2(x0 + 0.5 * h, y + 0.5 * l2)
        k4 = h * f1(x0 + h, z + k3)
        l4 = h * f2(x0 + h, y + l3)
        y = y + one_six * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        z = z + one_six * (l1 + 2.0 * l2 + 2.0 * l3 + l4)
        x0 = x0 + h
        Y[i] = y
        Z[i] = z
    return Y, Z


def plot(x, xn, y1, y2, true_y1, true_y2, h_1):
    xnew = np.linspace(x, xn, int((xn - x) / h_1 + 1))
    '''
    _f = []
    for el in xnew:
        _f.append(f[0]*el**2 + f[1]*el + f[2])
    '''
    #linestyle = 'dashed'
    plt.plot(xnew, y1, label="y_1(x)")
    #plt.plot(xnew, _f, linestyle = 'dashed', label="Многочлен 2ой степени
    # по критерию МНК")
    plt.plot(xnew, y2, label="y_2(x)")
    plt.plot(xnew, true_y1, linestyle = 'dashed', label="y_1(x) (odenit)")
    plt.plot(xnew, true_y2, linestyle = 'dashed', label="y_2(x) (odenit)")
    plt.legend()
    plt.grid(True)
    plt.show()


def vector(x, y, n):
    arr = [None] * n
    for i in range(n):
        arr[i] = sum(np.power(x, i) * y)
    return arr


def matrix(x, y, n):
    M = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            M[i][j] = sum(np.power(x, i+j))
    return list(reversed(np.linalg.solve(M, vector(x, y, n))))


def point(func, p):
    return np.polyval(func, p)


def mnk(x, y, f):
    return sum(np.power(y - point(f, x), 2))


def print_table(x, y, f):
    for i, j in zip(x, y):
        print(f'{f}({round(i, 3)}) = ', j)


def pend(y, t):
    y1, y2 = y
    dydt = [t / np.sqrt(1.0 + np.power(t, 2) + np.power(y2, 2)),
    y1 / np.sqrt(1.0 + np.power(t, 2) + np.power(y1, 2))]
    return dydt


if __name__ == '__main__':
    y_0 = 0.2
    z_0 = 0.0
    x_0 = -1.0
    b = 1.0
    h1 = 0.1
    X = np.arange(x_0, b+h1, h1, dtype=float)
    print(f'h = {h1}:')
    res, _res = runge_kutta(x_0, y_0, b, z_0, h1)
    res_half1, res_half2 = runge_kutta(x_0, y_0, b, z_0, 0.05)
    print_table(X, res, 'y1')
    print(f'h = {h1}:')
    print_table(X, _res, 'y2')
    f2 = matrix(X, res, 3)
    sol = odeint(pend, [y_0, z_0], np.arange(x_0, b+h1, h1))
    sol1 = sol[:, 0]
    sol2 = sol[:, 1]
    print(sol1)
    print(sol2)
    print("Многочлен 2-ой степени:", f2[0], "* x^2 +", f2[1], "* x +", f2[2])
    print("Квадратичное отклонение 2ой степени:", mnk(X, res, f2))
    err1 = []
    err2 = []
    new_res1 = []
    new_res2 = []
    for i in range(len(res_half1)):
        if i % 2 == 0:
            new_res1.append(res_half1[i])
            new_res2.append(res_half2[i])
    for el1, el2 in zip(res, new_res1):
        err1.append(abs(el2 - el1))
    print(sum(err1)/15.0)
    for el1, el2 in zip(_res, new_res2):
        err2.append(abs(el2 - el1))
    print(sum(err2)/15.0)
    plot(x_0, b, res, _res, sol1, sol2, h1)
