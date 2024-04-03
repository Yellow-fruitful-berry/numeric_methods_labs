import numpy as np
import matplotlib.pyplot as plt

# Функция, которая вычисляет значение интеграла методом трапеций
def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    nodes = np.linspace(a, b, n+1)
    integral = 0.5 * (f(a) + f(b))
    integral += np.sum(f(nodes[1:-1]))
    integral *= h
    return integral

# Определение функций K(x, t) и f(x)
def K(x, t):
    return x**2 * np.exp(x * t)

def f(x):
    return 1 - x * (np.exp(x) - np.exp(-x))

# Интервал интегрирования
a, b = -1, 1

# Количество узлов для численного интегрирования
n = 100

# Создание равномерной сетки узлов
x_values = np.linspace(a, b, n+1)

# Вычисление значения правой части уравнения для каждого узла
f_values = f(x_values)

# Матрица коэффициентов для системы уравнений
A = np.zeros((n+1, n+1))

for i in range(n+1):
    for j in range(n+1):
        A[i, j] = -trapezoidal_rule(lambda t: K(x_values[i], t), a, b, n)

# Добавление диагональных элементов
A += np.eye(n+1)

# Решение системы уравнений
u = np.linalg.solve(A, f_values)

# Построение графика
plt.plot(x_values, u, label='Numerical Solution')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Solution of the Fredholm Integral Equation')
plt.grid(True)
plt.legend()
plt.show()
