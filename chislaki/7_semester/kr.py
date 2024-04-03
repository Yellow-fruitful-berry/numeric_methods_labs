import numpy as np
from matplotlib import pyplot as plt
import scipy


def f1(x, y2):
    return x/np.sqrt(1+x*x+y2*y2)


def f2(x, y1):
    return y1/np.sqrt(1+x*x+y1*y1)


# function that returns dy/dt
def dydt(U, x):
    y2, y1 = U

    dydt1 = x/np.sqrt(1+x*x+y2*y2)
    dydt2 = y1/np.sqrt(1+x*x+y1*y1)

    return [dydt2, dydt1]


def real_f():
    # y1(-1)=0.2
    # y2(-1)=0

    y0 = [0, 0.2]

    # solution for range x = [-1, 1]

    h = 0.1
    t = np.arange(-1, 1+h, h)

    return scipy.integrate.odeint(dydt, y0, t)


def runge(x, y, h):
    k1 = h * f1(x, y)
    l1 = h * f2(x, y)

    k2 = h * f1(x + h*0.5, y + 0.5 * k1)
    l2 = h * f2(x + h*0.5, y + 0.5 * k1)

    k3 = h * f1(x + h*0.5, y + 0.5 * k2)
    l3 = h * f2(x + 0.5*h, y + 0.5 * k2)

    k4 = h * f1(x + h, y + k3)
    l4 = h * f2(x + h, y + k3)

    K = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    L = (l1 + 2 * l2 + 2 * l3 + l4) / 6
    return K, L


def least_squares(X, Y, tol=3):
    """
    Find least squares fit for coefficients of X given Y
        :param X: The input parameters
        :param Y: The output parameters or labels

        :return: The coefficients of X
                 including the constant for X^0
    """
    # Section 1: If X and/or Y are 1D arrays, make them 2D
    if not isinstance (X[0], list):
        X = [X]
    if not isinstance (type (Y[0]), list):
        Y = [Y]

    # Section 2: Make sure we have more rows than columns
    #            This is related to section 1
    if len (X) < len (X[0]):
        X = np.transpose (X)
    if len (Y) < len (Y[0]):
        Y = np.transpose (Y)

    # Section 3: Add the column to X for the X^0, or
    #            for the Y intercept
    for i in range (len (X)):
        X[i].append (1)

    # Section 4: Perform Least Squares Steps
    AT = np.transpose(X)
    ATA = np.matmul(AT, X)
    ATB = np.matmul(AT, Y)
    coefs = np.linalg(ATA, ATB, tol=tol)

    return coefs

def runge_output(a, b, h_, y1_0, y2_0):
    h = h_
    x_arr = np.arange(a, b + h, h)
    y_arr = [y1_0]
    z_arr = [y2_0]

    for i in range (1, len (x_arr)):
        K = runge (x_arr[i - 1], y_arr[i - 1], h)[0]
        L = runge (x_arr[i - 1], y_arr[i - 1], h)[1]

        y_arr.append (y_arr[i - 1] + K)
        z_arr.append (z_arr[i - 1] + L)

    plt.plot (x_arr, y_arr, label=('Numeric method y1, h=' + str(h)))
    plt.plot (x_arr, z_arr, label=('Numeric method y2, h=' + str(h)))
    return y_arr


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
y1_arr_h_01 = runge_output(a=-1, b=1, h_=0.1, y1_0=0.2, y2_0=0.0)
y1_arr_h_001 = runge_output(a=-1, b=1, h_=0.01, y1_0=0.2, y2_0=0.0)


h = 0.1
t_span = np.arange(-1, 1+h, h)

# Odeint
solution = real_f()

grid = (t_span, solution[:, 0])


pol = matrix(t_span, solution[:, 1], 3)

print("Polynomial coefficients:", pol)
p = np.poly1d(pol)

poly_y = []
sq_error = 0.0
error = 0.0
for i in range(len(t_span)):
    poly_y.append(p(t_span[i]))
    sq_error += (p(t_span[i]) - solution[:, 1][i]) ** 2
    error += abs(p (t_span[i]) - solution[:, 1][i])

plt.plot(t_span, poly_y, label='Approximation')
print("Square Error:", sq_error)
print("Error:", error)


# plot
plt.plot(t_span, solution[:, 1], label='y1')
plt.plot(t_span, solution[:, 0], label='y2')
plt.legend()
plt.xlabel('time')
leg = plt.legend()
plt.show()
