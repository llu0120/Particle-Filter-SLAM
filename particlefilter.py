#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 18:49:43 2019

@author: LuLienHsi
"""
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from load_data import encoder, imu, lidar
from map_utils import mapCorrelation
from utils_function import Body2World, MapInitialize, lamda2Binary, mapping, compute_motion_model, motion_difference, softmax

encoder = encoder()
encoder_ticks = encoder.encoder_counts[:,11:4902]  # 4 x n encoder counts
encoder_time = encoder.encoder_stamps[11:4902] # encoder time stamps

imu = imu()
imu_yaw = imu.imu_angular_velocity[2] 
imu_time = imu.imu_stamps

x_t,y_t,theta_t,velocity, angular_velocity, imu_time = compute_motion_model(encoder_ticks,encoder_time,imu_yaw,imu_time)


#%%
#Initialize MAP
MAP = MapInitialize()
Map = MAP['map']
resolution = MAP['res']
l = 4
x_im, y_im = np.arange(-40,40), np.arange(-40,40) 
xdif, ydif = np.arange(-resolution*l, resolution*l, resolution),  np.arange(-resolution*l, resolution*l, resolution)

#Initialize Particles
N = 100
particle_state = np.zeros((N,3))
weight = np.ones(N)/N
best_particle_state = np.zeros((3,))

#bigger noise for sampling particles and smaller noise for motion 
noise_particle = np.array([0.1, 0.1, 0.1*np.pi/180])
noise_motion = np.array([0.01, 0.01, 0.01*np.pi/180])

#intialize particle state 
for p in range(N):
    noise = noise_particle*np.random.randn(1,3)
    particle_state[p] = [x_t[0],y_t[0],theta_t[0]] + noise
    
#Initialize lidar ranges
lidar = lidar()
l_ranges = lidar.lidar_ranges 
l_angles = np.arange(-135,135.25,0.25)*np.pi/180.0

#Initialize logodds map  
lamda = np.zeros((1601,1601))
best_trajectory=[]

#%%
for i in range(len(x_t)):
    print(i)
    if i == len(x_t)-1: 
        break
    
    #im is the logodds map that need to plug into mapCorrelation
    im = lamda2Binary(lamda)
    
    #Prediction Step: predict the next pose of each particles     
    for p in range(N): 
        noise = noise_motion*np.random.randn(1,3)
        particle_state[p] = particle_state[p] + noise
        dtheta, dx, dy = motion_difference(velocity[i+1], angular_velocity[i+1], theta_t[i], encoder_time[i+1]-encoder_time[i], imu_time[i+1]-imu_time[i])
        
        particle_state[p,:] = [particle_state[p,0]+dx, particle_state[p,1]+dy, particle_state[p,2]+dtheta]
        
    print(particle_state[0],particle_state[1],particle_state[2])
    particle_state[:,2] %= 2*np.pi
    
    #New observation(Lidar measurement)
    next_scan = l_ranges[:,i]
    
    #Remove too far or too near scan 
    good = np.logical_and((next_scan < 30),(next_scan > 0.1))
    valid_l_ranges = next_scan[good]
    valid_l_angles = l_angles[good]
    
    xs0 = valid_l_ranges*np.cos(valid_l_angles) + 0.29833 
    ys0 = valid_l_ranges*np.sin(valid_l_angles)
    ScaninWorld = Body2World(xs0,ys0,theta_t[i])

    vp = np.vstack((ScaninWorld[0],ScaninWorld[1]))
    
    #Correlation
    max_correlation = []
    for p in range(N): 
        particle_cor_x, particle_cor_y = xdif + particle_state[p,0] , ydif + particle_state[p,1]
        map_correlation = mapCorrelation(im,x_im,y_im,vp,particle_cor_x,particle_cor_y)
        index = np.argmax(map_correlation)
        max_correlation.append(np.max(map_correlation))
    max_correlation = weight * np.array(max_correlation)
    weight = softmax(max_correlation)
    best_particle_index = np.argmax(weight)
    best_particle_state = particle_state[best_particle_index].copy()
    
    best_trajectory.append(particle_state[best_particle_index])
        
    lamda = mapping(best_particle_state[0], best_particle_state[1], best_particle_state[2], lamda, next_scan, x_t[i],y_t[i]) 

          


#%% 
#Transform log odds map to the binary map 
lamda = lamda2Binary(lamda)     
fig = plt.figure()
plt.imshow(lamda,cmap="gray");  






















