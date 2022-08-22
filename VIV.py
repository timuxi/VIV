import os
import time

import pandas as pd
import sympy
from sympy import *
import numpy as np
import matplotlib.pyplot as plt

# globals
from sympy.physics.quantum.identitysearch import scipy

import KF_basic
from VIVUtil import D_compute, h_compute, re_hilbert, hilbert_trans, read_data, sin_wave, integrate, wgn, wgn_new, \
    re_hilbert_new

t = 264
hz = 100  # 频率
x = np.arange(0, t, 1 / hz)  # 离散时间坐标轴
n = hz  # 采样框架
N = n * 1000  # 计算框架
h_n = 50
h_N = 2000
q = 0.98  # 高通滤波参数
w = 1 / hz
computing_frame = np.zeros(N)  # 计算框架,初始化有N个0的数组
A = np.transpose([0, 0])  # 矩阵A
xn = 0
# D矩阵
D = D_compute(N, hz=hz)
# h
h = h_compute(h_N)


def single_compute(sampling_frame):
    # 全局变量
    global computing_frame, xn, A
    # 删除计算框架中开头的n个元素
    computing_frame = computing_frame[n:N]
    # 在计算框架尾部插入采样框架
    computing_frame = np.append(computing_frame, sampling_frame)
    # 计算参数
    x_last = sum(computing_frame[0:n])
    tx_last = np.dot(
        computing_frame[0:n],
        np.transpose(np.arange(w, (n + 1) * w, w))
    )
    x_current = sum(computing_frame[N - n:N])
    tx_current = np.dot(
        computing_frame[N - n:N],
        np.transpose(np.arange((N - n + 1) * w, (N + 1) * w, w))
    )
    # 计算xn和A矩阵
    xn = xn - x_last + x_current
    B = np.transpose([x_last - x_current, tx_last + xn * n * w - tx_current])
    A = A - np.dot(D, B)

    # 校准基线
    for j in range(n):
        sampling_frame[j] -= (A[0] + A[1] * (N - n + j + 1) * w)

    return sampling_frame


def full_compute(data):
    global xn, computing_frame
    data_correct = []
    xn = 0
    computing_frame = np.zeros(N)

    for j in range(t):
        data_correct = np.append(data_correct, single_compute(data[j * n:(j + 1) * n]))
    # 高通滤波
    y = np.zeros(data_correct.size)
    for j in range(data_correct.size):
        if j == 0:
            pre_x = data_correct[j]
            pre_y = data_correct[j]
            y[j] = data_correct[j]
        else:
            y[j] = (1 + q) / 2 * (data_correct[j] - pre_x) + q * pre_y
            pre_x = data_correct[j]
            pre_y = y[j]

    return y


if __name__ == '__main__':
    fig, axs = plt.subplots(2, 4)

    # # # 读取数据
    data = read_data('acc.txt', t, hz, col=0)
    # data = sin_wave(1, 0.4536*np.pi, hz, t)
    #
    # 绘制初始加速度数据图像
    axs[0, 0].plot(x, data)
    axs[0, 0].set_title('acc_origin')
    #
    # 添加噪声
    # data += wgn(data, 20)
    axs[0, 1].plot(x, data)
    axs[0, 1].set_title('acc_noise')
    # 校准加速度
    now = time.time()
    acc = full_compute(data)
    print("time = " )
    print(time.time()- now)
    # 加速度积分
    velocity = integrate(acc, t, hz)
    print("time = ")
    print(time.time()- now)
    # 校准速度
    velocity = full_compute(velocity)
    # 绘制速度数据图像
    axs[0, 2].plot(x, velocity)
    axs[0, 2].set_title('velocity')
    # 速度积分
    dis = integrate(velocity, t, hz)
    # 校准位移
    dis = full_compute(dis)
    axs[0, 3].plot(x, dis)
    gt = sin_wave(-1 / (0.4536 * np.pi * 0.4536 * np.pi), 0.4536 * np.pi, hz, t)
    axs[0, 3].set_title('dis')
    # axs[0,3].plot(x,gt)
    # axs[1, 2].plot(x, abs(gt-dis))
    axs[1, 2].set_title('dis_err')
    # test1 = integrate(data, t, hz)
    # test2 = integrate(test1, t, hz)
    # axs[0, 3].plot(x, test2)
    #
    # # # 递推希尔伯特
    re_hilbert_new(dis, h=h, h_n=h_n, h_N=h_N, index1=1, index2=0, title='dis',axs=axs)
    # 原始希尔伯特转换
    # hilbert_trans(dis, 1, 2, 'dis', x,t,hz,axs)
    # # 绘制加速度的复域
    # hilbert_trans(acc, axs,1, 1, 'acc',x,t,hz)
    print("time = ")
    print(time.time()- now)
    plt.show()
