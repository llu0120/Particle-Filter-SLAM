#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:43:35 2019

@author: LuLienHsi
"""
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import os
os.chdir(r'/Users/LuLienHsi/Desktop/UCSD_Documents/2019_Winter/ECE276A_Sensing&EstimationRobotics/ECE276A_HW2/data') #进入指定的目录
from load_data import encoder,imu, lidar
from utils_function import Body2World, MapInitialize, xy2map, lamda2Binary, mapping, compute_motion_model

#%% motion model

encoder = encoder()
encoder_ticks = encoder.encoder_counts[:,11:4902]  # 4 x n encoder counts
encoder_time = encoder.encoder_stamps[11:4902] # encoder time stamps

imu = imu()
imu_yaw = imu.imu_angular_velocity[2] 
imu_time = imu.imu_stamps

x_t,y_t,theta_t,velocity, angular_velocity ,t= compute_motion_model(encoder_ticks,encoder_time,imu_yaw,imu_time)
#plt.scatter(x_t,y_t)
#plt.show()
#t_axis = np.arange(0,4891,1)
#plt.scatter(t_axis,theta_t)
#plt.show()

#%%
#Initialize MAP
MAP = MapInitialize()

#Initialize Particles
N = 1
particle_x = np.zeros(N) 
particle_y = np.zeros(N) 
particle_theta = np.zeros(N)
weight = np.ones(N)/N

best_particle_state =[]


#Initialize lidar ranges
lidar = lidar()
l_ranges = lidar.lidar_ranges 
l_angles = np.arange(-135,135.25,0.25)*np.pi/180.0

xorigin = particle_x[0]
yorigin = particle_y[0] 

#Initialize logodds map  
lamda = np.zeros((1601,1601))
next_scan = l_ranges[:,0]
#%%

for i in range(len(x_t)):
    print(i)
    if i == len(x_t)-1: 
        break
   
    lamda = mapping(particle_x[0], particle_y[0], particle_theta[0], lamda, next_scan, x_t[i],y_t[i]) 
    #Predict the next pose of the particle     
    diffx = x_t[i+1] - x_t[i]
    diffy = y_t[i+1] - y_t[i]
    difftheta = theta_t[i+1] - theta_t[i]

    particle_x[0] += diffx 
    particle_y[0] += diffy 
    particle_theta[0] += difftheta

    #New Lidar measurement 
    next_scan = l_ranges[:,i+1]


#%%    
lamda = lamda2Binary(lamda)     
fig = plt.figure()
plt.imshow(lamda,cmap="gray");    
#%%    
countplus = 0 
countminus = 0 
count = 0
for i in range(len(lamda)):
    for j in range(len(lamda[0])):
        if lamda[i][j] > 0:
            countplus += 1 
        elif lamda[i][j] < 0: 
            countminus +=1
        else: 
            count += 1    