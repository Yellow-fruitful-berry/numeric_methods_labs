# y = x/(3x+4)^3
import math

import numpy.linalg
import scipy.interpolate
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x/np.power((3*x+4), 3)


a = -1
b = 1
h = 0.5

l_border = a
r_border = a + h
sum_l = 0
while r_border <= b:
    sum_l += f(l_border)*h
    l_border += h
    r_border += h
print("Left rectangles: ", sum_l)

l_border = a
r_border = a + h
sum_l = 0
while r_border <= b:
    sum_l += f(r_border)*h
    l_border += h
    r_border += h
print("Right rectangles: ", sum_l)

l_border = a
r_border = a + h
sum_l = 0
while r_border <= b:
    sum_l += f((l_border+r_border)/2)*h
    l_border += h
    r_border += h
print("Center rectangles: ", sum_l)

l_border = a
r_border = a + h
sum_trapezoid = (f(a)+f(b))/2
while r_border < b:
    sum_trapezoid += f(r_border)
    l_border += h
    r_border += h
print("Trapezoid method: ", sum_trapezoid*h)


l_border = a
r_border = a + h
sum_simpson = f(a) + f(b)
while r_border <= b:
    sum_simpson += 4*(f((r_border+l_border)/2))
    l_border += h
    r_border += h
print("Simpson method: ", sum_simpson*h/6)



# -0.122448