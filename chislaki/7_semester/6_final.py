import numpy as np
import matplotlib.pyplot as plt

def p(x):
    return -4 * (x**2 + 3) / (x * (x**2 + 6))


def q(x):
    return 6 / (x**2 + 6)


def fin_diff(y_a, y_b, a, b, h):
    n = int((b - a) / h)
    x_values = np.linspace(a, b, n+1)
    A = np.zeros((n+1, n+1))
    B = np.zeros(n+1)
    A[0, 0] = 1.0
    A[0, 1] = -1.0
    A[n, n] = h-1
    A[n, n-1] = 1.0
    B[0] = -y_a*h
    B[n] = y_b*h
    for i in range(1, n):
        A[i, i] = -2.0 + q(x_values[i])*np.power(h, 2)
        A[i, i+1] = 1.0 + 0.5*p(x_values[i])*h
        A[i, i-1] = 1.0 - 0.5*p(x_values[i])*h
    y_values = np.linalg.solve(A, B)
    return x_values, y_values


def finite_difference_derivative(x, y, h):
    n = len(x)
    derivative = [(y[i+1] - y[i-1]) / (2 * h) if i > 0 and i < n-1 else 0 for i in range(n)]
    return derivative


x0 = 0
xn = 4
ya = 0
yb = 26
h1 = 0.1
h2 = 0.01
x_res_h1, y_res_h1 = fin_diff(ya, yb, x0, xn, h1)
x_res_h2, y_res_h2 = fin_diff(ya, yb, x0, xn, h2)

#print("y1", y_res_h1)
#print("y2", y_res_h2)

# Вычисляем производные
derivative_h1 = finite_difference_derivative(x_res_h1, y_res_h1, h1)
derivative_h2 = finite_difference_derivative(x_res_h2, y_res_h2, h2)

plt

# print('В нуле:', derivative_h2[0])

# print(f'Разность в точке {x_res_h1[-1]} = {y_res_h2[-1] - derivative_h2[-2]}')

plt.title('Finite difference method')
plt.plot(x_res_h1, y_res_h1, label='y(x), h = 0.1')
plt.plot(x_res_h2, y_res_h2, label='y(x), h = 0.01')
plt.plot(x_res_h1[:-1], derivative_h1[:-1], label="y'(x)")
plt.legend()
plt.grid(True)
plt.show()
