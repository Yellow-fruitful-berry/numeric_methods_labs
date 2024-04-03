print("Input points")
x = []
y = []

for i in range(5):
    x.append(float(input()))
for i in range (5):
    y.append(float(input()))

print("Input evaluation point: ", end = '')
x_star = float(input())

print("First method: ", (y[2]-y[1])/(x[2]-x[1]))

print("Second method: ", (y[3]-y[2])/(x[3]-x[2]))

print("Third method: ", (y[3]-y[1])/((x[3]-x[1])))

print("Second derivative: ", (y[3]-2*y[2]+y[1])/pow((x[2]-x[1]), 2))

'''
0
0.1
0.2
0.3
0.4
1.0
1.1052
1.2214
1.3499
1.4918

0.2
'''
