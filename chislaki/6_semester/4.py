import math

import numpy.linalg
import scipy.interpolate
import matplotlib.pyplot as plt
import numpy as np


def spline(a, b, c, d, x, x_curr):
    return a + b*(x-x_curr) + c*(x-x_curr)**2 + d*(x-x_curr)**3


x = []
y = []


x = [0.0, 0.9, 1.8, 2.7, 3.6]
y = [0.0, 0.36892, 0.85408, 0.7856, 6.3138]
n = len(x)
x_star = 1.5

# n = 5
c_matrix = [[0 for i in range(n-2)] for j in range(n-2)]
right_side = [0 for i in range(n-2)]
#
# for i in range(1, n): # start from 1, i=0 is index 1
#     h_minus = 0
#     h_plus = 0
#     if(i != 0):
#         h_minus = x[i - 1+1] - x[i - 2+1]
#     h = x[i +1] - x[i - 1+1]
#     #if(i!=n-2):
#     h_plus = x[i + 1+1] - x[i +1]
#
#     if(i != 0):
#         c_matrix[i][i-1] = h_minus
#     c_matrix[i][i] = 2*(h_minus + h)
#     #if(i!=n-2):
#     c_matrix[i][i+1] = h
#
#     if(i == 0):
#         right_side[i] = 3*((y[2]-y[1])/h + (y[1] - y[0]))
#     else:
#         right_side[i] = 3 * ((y[i + 2] - y[i + 1]) / h + (y[i + 1] - y[i]))
#


for i in range(len(c_matrix)):
    print(c_matrix[i])


c_matrix = [
            [3.6, 0.9, 0],
            [0.9, 3.6, 0.9],
            [0, 0.9, 3.6],
            ]

right_side = [
    3*(y[2]-y[1])/0.9-3*(y[1]-y[0])/0.9,
    3*(y[3]-y[2])/0.9-3*(y[2]-y[1])/0.9,
    3*(y[4]-y[3])/0.9-3*(y[3]-y[2])/0.9
]

print("r_s: ", right_side)
c1 = list(np.linalg.solve(c_matrix, right_side))

c2 = [0]

c = c2 + c1

print("c: ", c)

a = [0 for i in range(n-1)]
b = [0 for i in range(n-1)]
d = [0 for i in range(n-1)]

for i in range(1, n):
    h_i = (x[i]-x[i-1])

    a[i-1] = y[i-1]
    if (i == n-1):
        b[i-1] = (y[i] - y[i - 1]) / h_i - 2 * h_i * c[i-1] / 3
    else:
        b[i-1] = (y[i]-y[i-1]) / h_i - h_i * (c[i]+2*c[i-1])/3

    if(i == n-1):
        d[i-1] = (-1)*c[i-1]/(3 * h_i)
    else:
        d[i-1] = (c[i] - c[i-1])/(3 * h_i)

print("a: ", a)
print("b: ", b)
print("d: ", d)

def S(x_curr, x_i, a_, b_, c_, d_):
    return a_ + b_*(x_curr-x_i) + c_*(x_curr-x_i)**2 + d_*(x_curr-x_i)**3


for i in range(1, n):
    amount = 100
    y_spline = []
    x_ = np.linspace(x[i-1], x[i], amount)
    # print (i, i + 1)
    for x_current in x_:
        y_spline.append(S(x_current, x[i-1], a[i-1], b[i-1], c[i-1], d[i-1]))
    plt.plot(x_, y_spline, 'r-')

for l in range(n):
    plt.plot(x[l], y[l], 'ro')

plt.plot(x, y, 'b-', label = "Original")
plt.legend()
plt.show()






# spline_coeffs = [[0] * 4 for _ in range(n)]

#
# # indexing here is upgraded by 1 instead of what is in the lecture because of range() behaviour
# #setting c_i
# # c = [0]
# # first c1 = 0, as curvature is 0 at that end of spline
# right_side = [0 for i in range(n)]
# c_matrix = [[0 for i in range(n)] for j in range(n)]
#
# for i in range(0, n-1): # we need to create a tridiagonal linear equation system
#     if(i == 0):
#         #h_i_minus = 0
#         h_i = 0
#         h_i_1 = x[i + 1] - x[i]
#     elif(i == 1):
#         #h_i_minus = 0
#         h_i = x[i] - x[i - 1]
#         h_i_1 = x[i + 1] - x[i]
#     else:
#         #h_i_minus = x[i-1] - x[i-2] # h_i - 1
#         h_i = x[i] - x[i-1]  # h_i
#         h_i_1 = x[i + 1] - x[i]  # h_i + 1
#
#
#     right_side[i] = 3 * ((y[i+1]-y[i])/h_i_1-(y[i]-y[i-1]))
#
#     c_matrix[i][i-1] = h_i
#     c_matrix[i][i] = 2*(h_i + h_i_1)
#     c_matrix[i][i+1] = h_i_1
#
# a = []
# b = []
# d = []
#
# c_matrix[n-1][n-1] = 0
#
# print(c_matrix)
# print(right_side)
# c = numpy.linalg.solve(c_matrix, right_side)
#
# for i in range(n):
#     if (i == 0):
#         h_i = 0
#         h_i_1 = x[i + 1] - x[i]
#     else:
#         h_i = x[i] - x[i - 1]  # h_i
#         h_i_1 = x[i + 1] - x[i]  # h_i + 1
#
#     # setting a_i
#     a[i] = y[i]
#
#     if(i==0):
#         b[i] = a[i]/h_i + 2*c[i]*h_i/3
#     else:
#         b[i] = (a[i]-a[i-1])/h_i + ((2*c[i]+c[i-1])/3)*h_i
#
#     if(i == 0):
#         d[i] = c[i]/(3*h_i)
#     else:
#         d[i] = (c[i] - c[i-1])/(3*h_i)
#
#
# for l in range(n):
#     plt.plot(x[l], spline(a[l], b[l], c[l], d[l], x_star, x[l]), 'r-')



