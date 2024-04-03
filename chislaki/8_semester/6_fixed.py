import numpy as np
import matplotlib.pyplot as plt


def K(xi, xj):
    return xi**2 * np.exp(xi * xj)


def f(x):
    return 1 - x * (np.exp(x) - np.exp(-x))


def FRED_II_RECT(a, b, h):
    x = np.arange(a, b + h, h)
    n = len(x)
    wt = 0.5
    wj = 1
    A = np.zeros(shape=(n, n))
    for i in range(n):
        A[i][0] = -h*wt*K(x[i], x[0])
        for j in range(1, n):
            A[i][j] = -h*wj*K(x[i], x[j])
        A[i][n-1] = -h*wt*K(x[i], x[n-1])
        A[i][i] += 1
    B = np.zeros(shape=(n, 1))
    for j in range(n):
        B[j][0] = f(x[j])
    y = np.linalg.solve(A, B)
    return y


a = -1
b = 1
h = 0.1

y = FRED_II_RECT(a, b, h)



def fi(i, x, t, y):
    return K(x, t[i]) * y[i][0]

x = np.arange(a, b + h, h)
integral = [0] * len(x)
for i in range(len(x)):
    for j in range(1, len(x)):
        integral[i] += 0.5 * (fi(j, x[i], x, y) + fi(j-1, x[i], x, y)) * h

print(integral)
y_new = [0] * len(x)
for i in range(len(x)):
    y_new[i] = y[i][0]
res = [0] * len(x)
for i in range(len(x)):
    res[i] = y[i] - integral[i]

plt.plot(x, y)
plt.grid()
#plt.plot(x, res)
#plt.plot(x, integral)
plt.show()

plt.grid()
plt.plot(x, res)

plt.grid()
plt.plot(x, integral)

sum_nevyaz = 0
for i in range(len(y)):
    sum_nevyaz += abs(1 - i * 0.1 * (np.exp(0.1 * i) - np.exp(-0.1*i)) - y[i]+integral[i])*0.00001
    print((1 - i * 0.1 * (np.exp(0.1 * i) - np.exp(-0.1*i)) - y[i]+integral[i])*0.00001)

# print('integral:', integral)
print(sum_nevyaz)
