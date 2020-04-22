#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 11:11:35 2019

@author: LuLienHsi
"""

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import os
from load_data import kinect
from utils_function import xy2map, pixel2optical, CameraRotation, pixelBody2World



#%%
best_particle_state = np.load('/saved_npy/best_state.npy')
backgroundmap = np.load('/saved_npy/lamda.npy')
rgbmap = np.zeros((1601,1601,3))
for i in range(len(backgroundmap)):
    for j in range(len(backgroundmap[0])):
        if backgroundmap[i][j] == 0: 
            rgbmap[i][j] = [192/255,192/255,192/255] 
        elif backgroundmap[i][j] == 1: #obstacle
            rgbmap[i][j] = [255/255,255/255,255/255]
        else:
            rgbmap[i][j] = [0,0,0]
#%%            
figure = plt.figure()
plt.scatter(best_particle_state[:,0],best_particle_state[:,1],s = 10 , alpha=0.5)
plt.imshow()
            
#%%
encoder_time_stamps = np.load('/data/encoder_time.npy')
kinect = kinect()
disp_stamps = kinect.disp_stamps
rgb_stamps = kinect.rgb_stamps
dis_index_list = []
for i in range(len(rgb_stamps)):
   diff = list(abs(disp_stamps - rgb_stamps[i]))
   diff_min = np.min(diff)
   index = diff.index(diff_min)
   dis_index_list.append(index)
disp_time_scaled = disp_stamps[dis_index_list]

index_list = []
for i in range(len(rgb_stamps)):
   diff = list(abs(encoder_time_stamps - rgb_stamps[i]))
   diff_min = np.min(diff)
   index = diff.index(diff_min)
   index_list.append(index) 
encoder_time_scaled = encoder_time_stamps[index_list]

best_particle_state = np.concatenate((best_particle_state,best_particle_state[[4889]]),axis=0)
best_particle_state_scaled = best_particle_state[index_list,:]

    
    
#%%
    #len(best_particle_state_scaled)
for i in range(len(best_particle_state_scaled)):
    print(i)
    os.chdir(r'/dataRGBD/RGB20') #go into directory 
    rgbd = plt.imread('rgb20_{}.png'.format(i+1))
    os.chdir(r'/dataRGBD/Disparity20') #go into directory 
    disparity = plt.imread('disparity20_{}.png'.format(dis_index_list[i]))
#    rgb_optical = np.zeros((height,width), dtype = [('x','f8'),('y','f8'),('z','f8')])
    dd = (-0.00304*disparity + 3.31)
    depth = 1.03/dd
    rgbi = np.rint((dd*(-4.5)*1750.46+19276.0)/585.051 + np.arange(0,rgbd.shape[0]).reshape([-1,1])*526.37/585.051)
    rgbj = np.rint(np.tile((np.arange(0,rgbd.shape[1]).reshape([1,-1]) * 526.37 + 16662)/585.051,(rgbd.shape[0],1)))
    flat_depth = depth.flatten()
    flat_rgbi = rgbi.flatten()
    flat_rgbj = rgbj.flatten()
    flat_ones = np.ones((len(flat_rgbi)))
    pixeluv1 = np.vstack((flat_rgbi,flat_rgbj,flat_ones))
    pixelinoptical = pixel2optical(10*pixeluv1,flat_depth)
    opticalinbody = CameraRotation(pixelinoptical)+np.array([[0.18],[0.005],[0.36]])
    pixelinWorld = pixelBody2World(opticalinbody,best_particle_state_scaled[i,2])
    
    
    temp_index = np.where(pixelinWorld[2,:] < 3.4)
    index_i = flat_rgbi[temp_index].astype('int')
    index_j = flat_rgbj[temp_index].astype('int')
    map_x,map_y = xy2map(pixelinWorld[0,temp_index] + best_particle_state_scaled[i][0]), xy2map(pixelinWorld[1,temp_index] + best_particle_state_scaled[i][1])
    rgbmap[map_x,map_y,:] = rgbd[index_i,index_j,:]

    
    
    
#%%          
fig = plt.figure()
plt.imshow(rgbmap);  
            
            
            
