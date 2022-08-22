import os
import random

import numpy as np
from scipy.fft import fft
import pandas as pd
import sympy
from sympy import *
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.signal import hilbert


# 显示频域谱线
from sympy.parsing.sympy_parser import null


def disPlaySpectrum(data, fs, axs, index1, index2, title):
    """
    data : 输入数据
    fs : 采样频率
    index1: 绘图x轴坐标
    index2: 绘图y轴坐标
    """
    N = data.size

    ft = fft(data) / N * 2
    ft = ft[0:N // 2]
    f = np.arange(0, N, 1) * fs / N
    f = f[0:N // 2]

    axs[index1, index2].plot(f, abs(ft))
    axs[index1, index2].set_title(title + "_fft")
    axs[index1, index2].set_xlabel('Hz')
    axs[index1, index2].set_ylabel('m/s^2')


# 添加噪声
def wgn_new(data):
    for i in range(0, data.size, 10):
        noise_index = random.randint(0, 9)
        data[i+noise_index] += random.uniform(1,2)
    return data


def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


# 生成正弦信号
def sin_wave(A, f, fs, t, phi=0):
    """
      A:振幅
      f:信号频率
      fs:采样频率
      t:时间长度
      phi:相位(可选，默认为0)
    """
    # 若时间序列长度为 t=1s,
    # 采样频率 fs=1000 Hz, 则采样时间间隔 Ts=1/fs=0.001s
    # 对于时间序列采样点个数为 n=t/Ts=1/0.001=1000, 即有1000个点,每个点间隔为 Ts
    Ts = 1 / fs
    n = t / Ts
    n = np.arange(n)
    y = A * np.sin(f * n * Ts + phi * (np.pi / 180))
    return y


def cos_wave(A, f, fs, t, phi=0):
    """
      A:振幅
      f:信号频率
      fs:采样频率
      t:时间长度
      phi:相位(可选，默认为0)
    """
    # 若时间序列长度为 t=1s,
    # 采样频率 fs=1000 Hz, 则采样时间间隔 Ts=1/fs=0.001s
    # 对于时间序列采样点个数为 n=t/Ts=1/0.001=1000, 即有1000个点,每个点间隔为 Ts
    Ts = 1 / fs
    n = t / Ts
    n = np.arange(n)
    y = A * np.cos(f * n * Ts + phi * (np.pi / 180))
    return y

# 积分
def integrate(data, t, hz):
    """
    data:    需要积分的数据
    t:       时间
    hz:      采样频率
    """
    x = np.arange(0, t, 1/hz)  # 离散时间坐标轴
    print(len(x))
    print(len(data))
    data_integrate = scipy.integrate.cumtrapz(data, x)
    data_integrate = np.append(data_integrate,data_integrate[len(data_integrate)-1])
    return data_integrate


def D_compute(compute_Frame_N, hz=50):
    """
    compute_Frame_N: 计算框架长度
    hz:      采样频率,默认为50hz
    """
    temp1 = 0
    temp2 = 0
    for i in range(compute_Frame_N):
        temp1 = temp1 + i / hz
        temp2 = temp2 + i / hz ** 2
    D = np.linalg.inv(([compute_Frame_N, temp1], [temp1, temp2]))
    return D


def h_compute(h_N):
    h = np.zeros(h_N)
    for i in range(1, h_N):
        h[i] = 2 / h_N * (np.sin(i * np.pi / 2) ** 2) / np.tan(np.pi * i / h_N)
    return h


# 递归希尔伯特计算
def re_hilbert_compute(u, h, h_N):
    ans = 0
    for m in range(1, h_N):
        ans += (h[h_N - m] - h[h_N - m - 1]) * u[m]
    return ans


def ration_compute(u, v):
    A_t = np.sqrt(u ** 2 + v ** 2)
    A_max = np.max(A_t)
    A_min = np.min(A_t)
    return A_min / A_max



# 递推希尔伯特
def re_hilbert(u, h, h_n, h_N, index1, index2, title,axs=null):
    # 最后返回的虚部
    v = np.zeros(u.size)
    # 旧的采样框架
    index_old = 0
    block_old = u[index_old: index_old + h_n]
    # 新的采样框架
    index_new = h_N
    block_new = u[index_new: index_new + h_n]
    # 初始化计算框架
    h_compute_frame = u[0: h_N]
    v[0:h_N] = np.imag(scipy.signal.hilbert(u[0: h_N]))
    for j in range(h_N, u.size, h_n):
        temp = re_hilbert_compute(h_compute_frame, h, h_N)

        for k in range(0, h_n):
            v[j+k] = (v[j-1+k] + h[0]*block_new[k] - h[h_N - 1]*block_old[k] + temp)

        # 加入新的采样框架
        h_compute_frame = np.append(h_compute_frame, block_new)
        # 舍弃旧的采样框架
        h_compute_frame = h_compute_frame[h_n: h_n+h_N]
        # 往后推h_n
        index_old += h_n
        block_old = u[index_old: index_old + h_n]
        index_new += h_n
        block_new = u[index_new: index_new + h_n]

    # 绘制希尔波特图
    axs[index1, index2].plot(u)
    axs[index1, index2].plot(v)
    axs[index1, index2].set_title(title + '_re_hilbert')
    axs[index1, index2 + 1].plot(u[1000:u.size-1000], v[1000:u.size-1000])
    axs[index1, index2 + 1].set_title(title + '_re_hilbert')


# 递推希尔伯特
def re_hilbert_new(u, h, h_n, h_N, index1, index2, title,axs=null):
    # 最后返回的虚部
    v = np.zeros(u.size)
    # 初始化计算框架
    h_compute_frame = u[0: h_N]
    v[0:h_N] = np.imag(scipy.signal.hilbert(u[0: h_N]))
    for j in range(h_N, u.size):
        temp = re_hilbert_compute(h_compute_frame, h, h_N)
        v[j] = (v[j-1] + h[0]*h_compute_frame[h_N-1] - h[h_N - 1]*h_compute_frame[0] + temp)
        if j != u.size-1 :
            # 加入新的采样框架
            h_compute_frame = np.append(h_compute_frame, u[j + 1])
            # 舍弃旧的采样框架
            h_compute_frame = h_compute_frame[1: 1 + h_N]
    # 绘制希尔波特图
    axs[index1, index2].plot(u)
    axs[index1, index2].plot(v)
    axs[index1, index2].set_title(title + '_re_hilbert')
    axs[index1, index2 + 1].plot(u[4000:u.size-1000], v[4000:u.size-1000])
    axs[index1, index2 + 1].set_title(title + '_re_hilbert')

# 希尔伯特转化
def hilbert_trans(u, index1, index2, title, x, t, hz, axs=null):
    v = scipy.signal.hilbert(u)
    hx = np.imag(v)
    p = t*hz
    r = np.zeros(u.size//p)
    index = 0
    for j in range(0,u.size,p):
        temp2 = ration_compute(u[j:j+p], hx[j:j+p])
        r[index] = temp2
        index += 1
    hx = np.imag(v)
    # 绘制希尔波特图
    # if axs !=null:
    axs[index1, index2].plot(u[60 * 50: u.size - 30 * 50], hx[60 * 50: u.size - 30 * 50])
    axs[index1, index2].set_title(title + '_hilbert_origin')
    axs[index1, index2 + 1].plot(r)
    axs[index1, index2 + 1].set_title(title + '_ration')


def read_data(_filename, t, hz, col=1):
    path = os.path.abspath('.//VIV_data_files') + '//' + _filename
    pd_read = pd.read_csv(path, header=None)
    pd_read = pd_read.values[:, col]
    pd_read = pd_read[0: t*hz]
    print('读取数据量: ', pd_read.size)
    return pd_read
