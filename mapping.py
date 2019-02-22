#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 15:20:54 2019

@author: LuLienHsi
"""
#%%
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import os
os.chdir(r'/Users/LuLienHsi/Desktop/UCSD_Documents/2019_Winter/ECE276A_Sensing&EstimationRobotics/ECE276A_HW2/data') #进入指定的目录

from load_data import lidar
from utils_function import mapping
#%%
countplus = 0 
countminus = 0 
count = 0
l = lidar()
l_ranges = l.lidar_ranges  
lamdaog = np.zeros((1601,1601))

#%%
updated_lamda = mapping(0,0,0,lamdaog,l_ranges[:,0],0,0)
for i in range(len(lamdaog)):
    for j in range(len(lamdaog[0])):
        if updated_lamda[i][j] > 0:
            countplus += 1 
        elif updated_lamda[i][j] < 0: 
            countminus +=1
        else: 
            count += 1
fig2 = plt.figure()
plt.imshow(updated_lamda,cmap="gray");           
