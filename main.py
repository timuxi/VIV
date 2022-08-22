# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:02:24 2022

@author: guoch
"""

import Kalmain as kf
import viv_basic as viv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    acc,gnss,al,gl = kf.get_data()
    acc = acc.tolist()
    gnss = gnss.tolist()
    if al != gl:
        if al>gl:
            fst = al/gl
            l = gl
        acc_align = []
        for i in range(gl):
            wa = int(fst*i)
            wb = int(wa + fst)
            w = acc[wa:wb]
            w = np.mean(w)
            acc_align.append(w)

    
    gnss_mean = np.mean(gnss)
    for i in range(gl):
         gnss[i]=gnss[i]-gnss_mean 
    
    dis = viv.freq_integ(acc_align, fst, 0.1, 1)
    # x = np.arange(0,l,1) 
    # plt.clf()
    # plt.plot(x,dis,'b')
    # plt.plot(x,gnss,'r')
    # plt.show()
    
    
    k = kf.KalmanFilter()
    k.a = dis[0]
    k.z = gnss[0]
    pred = list()
    pred_position = list()
    for i in range(l):
        k.a = dis[i]
        k.bkf()
        k.z = gnss[i]
        k.bkf()
        pred.append(k.NN)
        pred_position.append(np.float64(k.NN[0]))
        
    # x_axis = np.linspace(0,0.1,1200)
    err1 = []
    err2 = []
    for i in range(l):
        err_m = abs(gnss[i] - pred_position[i])
        err_m2 = abs(gnss[i] - dis[i])
        err1.append(err_m)
        err2.append(err_m2)
        
    viv.VIVdt3(dis, [1,0.2], [0.9,1]) 
    
    # viv.VIVdt3(pred_position, [1,0.2], [0.9,1])   
    
    # 画图    
    # x_axis = np.arange(0,l,1) 
    # print(gnss)
    # print(pred_position)
    # print(len(gnss),type(gnss[0]),len(gnss),type(pred_position[0]))
    # plt.clf()
    # fig = plt.figure(figsize=(8, 6), dpi=300)
    # plt.plot(x_axis, gnss, color = 'r', label = 'ob_position')
    # plt.plot(x_axis, pred_position, color = 'b', label = 'pred_position')
    # plt.legend(loc = 'best')
    # plt.plot(x_axis,err,'b')
    # plt.show()
    
    # fig, axs = plt.subplots(1,2,sharex=False, sharey=False,figsize = (8,4),dpi = 300)
    # plt.suptitle('ob pre error')
    # axs[0].plot(x_axis, dis, color = 'b', label = 'ob_position')
    # axs[0].plot(x_axis, pred_position, color = 'r', label = 'pred_position')
    # axs[0].set_title('ob pre')
    
    # axs[1].plot(x_axis,err2,'r')
    # axs[1].plot(x_axis,err1,'b')
    # axs[1].set_title('err')
   
    # plt.tight_layout()