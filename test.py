import random
from cmath import cos, pi, sin

import matplotlib.pyplot as plt
import numpy as np


from VIVUtil import sin_wave, integrate

a = random.randint(0, 4)


hz = 100
t = 20
x = np.arange(0, t, 1 / hz)  # 离散时间坐标轴




def sum_fun_xk(xk, func):
    return sum([func(each) for each in xk])

def integral(a, b, n, func):
    h = (b - a) / float(n)
    xk = [a + i * h for i in range(1, n)]
    return h / 2 * (func(a) + 2 * sum_fun_xk(xk, func) + func(b))

def test(data,hz,t):
    dt =1/hz
    a = data[0]
    arr = []
    sum = -1
    for i in range(1, hz*t-1):
        b = data[i]
        sum += (a+b)*dt/2
        arr.append(sum)
        a = b
    arr.append(arr[len(arr)-1])
    return arr


if __name__ == '__main__':
    print(np.sin( pi / 2))