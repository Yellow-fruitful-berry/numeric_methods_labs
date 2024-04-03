import math

import numpy.linalg
import scipy.interpolate
import matplotlib.pyplot as plt
import numpy as np

def PolyCoefficients(x, coeffs):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    o = len(coeffs)
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return y


x = []
y = []

x = [-0.9, 0.0, 0.9, 1.8, 2.7, 3.6]
y = [-0.36892, 0.0, 0.36892, 0.85408, 1.7856, 6.3138]
n = len(x)  # function node amount

for l in range(n):
    plt.plot(x[l], y[l], 'ro')


#k = 1  # polinomial degree
k_ = [1, 2, 3]

for k in k_:

    # reinitialisation of vectors and matrices for current k
    A = [[0] * (k + 1) for _ in range(k + 1)]
    # x = [0]*k
    B = [None] * (k + 1)

    for i in range(k + 1):
        for j in range(k + 1):
            A[i][j] = sum(np.power(x, i + j))

    for i in range(k + 1):
        B[i] = sum(np.power(x, i) * y)


    # print(A)
    # print(B)

    x_vector = numpy.linalg.solve(A, B)

    print(x_vector)

    error = 0
    res = [None] * n
    for m in range(n):
        res[m] = PolyCoefficients(x[m], x_vector)
        error += np.power(y[m] - res[m], 2)

    y_approx = []
    amount = 100
    x__ = np.linspace(min(x), max(x), amount)

    for m in range(amount):
        y_approx.append(PolyCoefficients(x__[m], x_vector))



    print("Mean squared error for k={}".format(k) , " is ", error/n)


    plt.plot (x__, y_approx, label = "k={}".format(k))
    #plt.plot(x, res, label = "k={}".format(k))


plt.plot(x, y, 'b-', label = "Original")
plt.legend()
plt.show()
