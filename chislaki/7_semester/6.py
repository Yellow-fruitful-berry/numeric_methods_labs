import numpy as np
import matplotlib.pyplot as plt


def p(x):
    return 0.0


def q(x):
    return -2.0 / (np.power(x, 2) + 1.0)


def fin_diff(y_a, y_b, a, b, h):
    n = int((b - a) / h)
    x = np.linspace(a, b, n)
    mA = np.zeros((n+1, n+1))
    mB = np.zeros(n+1)
    mA[0, 0] = 1.0
    mA[0, 1] = -1.0
    mA[n, n] = 1.0
    mB[0] = -y_a*h
    mB[n] = y_b
    for i in range(1, n):
        mA[i, i] = -2.0 + q(x[i])*np.power(h, 2)
        mA[i, i+1] = 1.0 + 0.5*p(x[i])*h
        mA[i, i-1] = 1.0 - 0.5*p(x[i])*h
    Yn = np.linalg.solve(mA, mB)
    #_Y = [(Yn[i] - Yn[i-1]) / h for i in range(1,n)]
    #print(_Y)
    return Yn


def plot(x, xn, y1, y2, h_1, h_2):
    XNEW1 = np.linspace(x, xn, len(y1))
    XNEW2 = np.linspace(x, xn, len(y2))
   # _XNEW1 = np.linspace(x, xn, len(_y1))
    #_XNEW2 = np.linspace(x, xn, len(_y2))
    plt.plot(XNEW1, y1, label=f'y(x) при h = {h_1}')
    plt.plot(XNEW2, y2, label=f'y(x) при h = {h_2}')
    #plt.title('Метод конечных разностей')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    x0 = 0.0
    xn = 1.0
    y0 = 2.0
    yn = 3.0 + 0.5*np.pi
    h1 = 0.1
    h2 = 0.01
    res = fin_diff(y0, yn, x0, xn, h1)
    _res = fin_diff(y0, yn, x0, xn, h2)
    print('h = 0.1')
    print(res)
    print('h = 0.01')
    print(_res)
    plot(x0, xn, res, _res, h1, h2)