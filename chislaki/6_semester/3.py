import math
import scipy.interpolate
import matplotlib.pyplot as plt
import numpy as np


def inverted_sign(value):
    if value < 0:
        return ' '
    else:
        return ' - '


print("Input degree:\n")
n = int(input())

coeffs = []
coeffs_y = []

print("Input coefficients:\n")
for i in range(n):
    x_i = float(input())
    coeffs.append(x_i)
    coeffs_y.append(math.pow(math.e, x_i))


lagrange_value = 0
print("Input point:\n")
x_star = float(input())
print("Real value: ", math.pow(math.e, x_star))

lagrange_coeffs = []

for i in range(n):
    nominator = 1
    denominator = 1
    for j in range(n):
        if(i != j):

            nominator *= x_star-coeffs[j]

            denominator *= coeffs[i] - coeffs[j]

    lagrange_value += coeffs_y[i] * nominator / denominator





print("Lagrange approximated value: ", lagrange_value)

newton_coeffs = []

def div_diff(x, y):
    n = len(x)
    if n == 1:
        return y[0]
    else:
        res = 0
        for j in range (n):
            den = 1
            for i in range (n):
                if (i != j):
                    den *= x[j] - x[i]
            res += y[j] / den
        return res


newton_value = 0
for i in range(n):
    row = 1
    for j in range(i):
        row *= x_star - coeffs[j]
    newton_value += row * div_diff(coeffs[:i+1], coeffs_y[:i+1])


print("Newton approximated value: ", newton_value)











x_parabola = np.array(coeffs) # array for x
y_parabola = np.array(coeffs_y) # array for y
plt.figure()
u = plt.plot(x_parabola,y_parabola,'ro') # plot the points
t = np.linspace(0, 1, len(x_parabola)) # parameter t to parametrize x and y
pxLagrange = scipy.interpolate.lagrange(t, x_parabola) # X(T)
pyLagrange = scipy.interpolate.lagrange(t, y_parabola) # Y(T)
n = 100
ts = np.linspace(t[0],t[-1],n)
xLagrange = pxLagrange(ts) # lagrange x coordinates
yLagrange = pyLagrange(ts) # lagrange y coordinates
plt.plot(xLagrange, yLagrange,'b-',label = "Polynomial")

plt.plot(xLagrange, np.exp(xLagrange),'r-',label = "Exponent")

plt.plot(xLagrange, np.exp(xLagrange),'r-',label = "Exponent")

plt.show()
# inputs:

'''

4


-2
-1
0
1

-0.5





'''





#up[-1] += '(x ' + inverted_sign(coeffs[j]) + coeffs[j]
            #down[-1] += '(' + coeffs[i] + inverted_sign(coeffs[j]) + coeffs[j]
            #if(j != n-1):
                #up[-1] += '*'
                #down[-1] += '*'
                #pass