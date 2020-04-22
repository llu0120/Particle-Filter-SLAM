#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:43:35 2019

@author: LuLienHsi
"""
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from load_data import encoder, imu, lidar
from utils_function import Body2World, MapInitialize, xy2map, lamda2Binary, mapping, compute_motion_model
from map_utils import bresenham2D

'''
Try the mapping process with the imu and wheel encoder data, instead of using the particle filter to do SLAM
'''
#%% motion model
encoder = encoder()
encoder_ticks = encoder.encoder_counts[:,11:4902]  # 4 x n encoder counts
encoder_time = encoder.encoder_stamps[11:4902] # encoder time stamps

imu = imu()
imu_yaw = imu.imu_angular_velocity[2] 
imu_time = imu.imu_stamps


#Plot only the motion of the imu and wheel encoder 
x_t,y_t,theta_t,velocity, angular_velocity ,t = compute_motion_model(encoder_ticks,encoder_time,imu_yaw,imu_time)
plt.scatter(x_t,y_t)
plt.show()
t_axis = np.arange(0,4891,1)
plt.scatter(t_axis,theta_t)
plt.show()

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

#world coordinate 
xorigin = particle_x[0]
yorigin = particle_y[0] 

#Initialize logodds map  
lamda = np.zeros((1601,1601))
next_scan = l_ranges[:,0]

lamda = lamda2Binary(lamda)     

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
    lamda = lamda2Binary(lamda)     
    fig = plt.figure()
    plt.imshow(lamda,cmap="gray");
    plt.scatter(particle_x[0]+800,particle_y[0]+800, s = 5)
    plt.show()
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
            



def mapping(x, y, theta, lamda_og, scan, x_t, y_t):#Input Lidar origin in Lidar frame x,y,theta
    # Initialize
    l_angles = np.arange(-135,135.25,0.25)*np.pi/180.0
    first_scan = scan
    
    # take valid indices
    good = np.logical_and((first_scan < 30),(first_scan > 0.1))
    valid_l_ranges = first_scan[good]
    valid_l_angles = l_angles[good]

    # init MAP
    MAP = MapInitialize()

    #size of map 801*801 
    #Lidar origin in coordinate
    xorigin = x + 0.29833 * np.cos(theta)
    yorigin = y + 0.29833 * np.sin(theta)
    
    
    
    # Compute lidar scan under world frame: ranges --> x,y --> Body2World
    xs0 = valid_l_ranges*np.cos(valid_l_angles) + 0.29833 
    ys0 = valid_l_ranges*np.sin(valid_l_angles)
    
    ScaninWorld = Body2World(xs0,ys0,theta)
    
    xs0 = ScaninWorld[0] + x_t
    ys0 = ScaninWorld[1] + y_t
    
    
    # convert from meters to cells
    xorigin_is = xy2map(xorigin)
    yorigin_is = xy2map(yorigin)
    xis = xy2map(xs0) 
    yis = xy2map(ys0) 
  
    # build an arbitrary map 
    indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
    MAP['map'][xis[indGood[0]],yis[indGood[0]]]=1

    # intialize lamda (log odds map) and use bresenham algorithm to detect how many grids that the lidar scan goes through
    lamda = lamda_og#np.zeros((801,801))
    log_odds= np.log(4)
    bmap=[]
    for i in range(len(xis)):
        bmap.append(bresenham2D(xorigin_is,yorigin_is,xis[i],yis[i]))
    #bmap = bmap
    xlast = []
    ylast = []
    for i in range(len(bmap)):
        xlast = int(bmap[i][0][-1])
        ylast = int(bmap[i][1][-1])
        lamda[xlast][ylast] += log_odds
        for j in range(len(bmap[i][0])):
            if bmap[i][0][j] != xlast and bmap[i][1][j] != ylast:
                xelse = int(bmap[i][0][j])
                yelse = int(bmap[i][1][j])
                lamda[xelse][yelse] -= log_odds
    
    lamda = lamda2Binary(lamda)               
    return lamda
















