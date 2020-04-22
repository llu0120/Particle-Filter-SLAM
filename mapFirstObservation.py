#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 15:20:54 2019

@author: LuLienHsi
"""

'''
This is a test for seeing the mapping process correctness
'''
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from load_data import lidar
from utils_function import mapping

countplus = 0 
countminus = 0 
count = 0
l = lidar()
l_ranges = l.lidar_ranges  

#Initialize the map with all 0(gray, unknown space)
lamdaog = np.zeros((1601,1601))
m, n = len(lamdaog), len(lamdaog[0])

updated_lamda = mapping(0,0,0,lamdaog,l_ranges[:,0],0,0)
for i in range(m):
    for j in range(n):
        if updated_lamda[i][j] > 0:
            countplus += 1 
        elif updated_lamda[i][j] < 0: 
            countminus +=1
        else: 
            count += 1
fig2 = plt.figure()
plt.imshow(updated_lamda,cmap="gray");           
