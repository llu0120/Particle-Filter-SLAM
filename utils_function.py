import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import os
os.chdir(r'/Users/LuLienHsi/Desktop/UCSD_Documents/2019_Winter/ECE276A_Sensing&EstimationRobotics/ECE276A_HW2/data') #进入指定的目录
from map_utils import bresenham2D
from numpy.linalg import inv



def Body2World(x,y,theta): 
    Transform_matrix = [[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]
    result = np.dot(Transform_matrix,np.transpose([x,y,0]))
    return result
    
       
def MapInitialize():
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -40  #meters
    MAP['ymin']  = -40
    MAP['xmax']  =  40
    MAP['ymax']  =  40 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) 
    return MAP

def xy2map(x):
    MAP = MapInitialize()
    xis = np.ceil((x - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    return xis

def lamda2Binary(lamda): 
    lamda[np.where(lamda < 0)]= -1
    lamda[np.where(lamda == 0)] = 0
    lamda[np.where(lamda > 0)] = 1
    return lamda
    
def mapping(x,y,theta,lamda_og,scan,x_t,y_t):#Input Lidar origin in Lidar frame x,y,theta
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
    # Compute lidar origin under world frame
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

def compute_motion_model(encoder_ticks,encoder_time,imu_yaw,imu_time):
    
    imu_time_scaled = []
    index_list=[]
    #make the timestamps of IMU and encoder equal 
    for i in range(len(encoder_time)):
       diff = list(abs(imu_time - encoder_time[i]))
       diff_min = np.min(diff)
       index = diff.index(diff_min)
       index_list.append(index)
    imu_time_scaled = imu_time[index_list]
    imu_yaw = imu_yaw[index_list]
    
    #linear velocity
    velocity = [0]
    right_dis = (encoder_ticks[0] + encoder_ticks[2])/2*0.0022
    left_dis =  (encoder_ticks[1] + encoder_ticks[3])/2*0.0022
    distance = (right_dis + left_dis)/2
    for i in range(len(encoder_time)):
        if i != 0:
            dt = encoder_time[i] - encoder_time[i-1]
            velocity.append(distance[i]/dt)      
            
            
    x_t,y_t,theta_t = [],[],[]
    
    for i in range(len(imu_time_scaled)):
        if i == 0: 
            theta_t.append(0 + imu_yaw[1]*(imu_time_scaled[1]-imu_time_scaled[0]))
        else: 
            theta_t.append(theta_t[i-1] + imu_yaw[i]*(imu_time_scaled[i]-imu_time_scaled[i-1]))
    
    for i in range(len(encoder_time)):
        if i == 0: 
            x_t.append(0)
            y_t.append(0)
        else: 
            dt = encoder_time[i]-encoder_time[i-1]
            xsin_term = np.sin(imu_yaw[i]*dt/2)
            xcos_term = np.cos(theta_t[i-1] + imu_yaw[i]*dt/2)
            x_t.append(x_t[i-1] + velocity[i]*dt*xsin_term*xcos_term/(imu_yaw[i]*dt/2))
    
            ysin1_term = np.sin(imu_yaw[i]*dt/2)
            ysin2_term = np.sin(theta_t[i-1] + imu_yaw[i]*dt/2)
            y_t.append(y_t[i-1] + velocity[i]*dt*ysin1_term*ysin2_term/(imu_yaw[i]*dt/2))
    angular_velocity = imu_yaw
    return x_t, y_t, theta_t, velocity, angular_velocity, imu_time_scaled

def motion_difference(velocity, omega, thetai, encoder_time_diff, imu_time_diff):
    if imu_time_diff == 0: 
        imu_time_diff = encoder_time_diff
    theta_diff = omega*imu_time_diff
    
    omega_term = omega * imu_time_diff/2 

    x_diff = velocity*  encoder_time_diff*(np.sin(omega_term)/omega_term)*np.cos(thetai + omega_term)
    y_diff = velocity*encoder_time_diff*(np.sin(omega_term)/omega_term)*np.sin(thetai + omega_term)
  
    return theta_diff, x_diff, y_diff
    
    
def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x / e_x.sum()

def pixel2optical(pixeluv1,flat_depth):
    K = [[585.05108211, 0, 242.94140713],
         [0, 585.05108211, 315.83800193],  
         [0, 0, 1]]
    canonical = flat_depth
    result = canonical*np.dot(inv(K), pixeluv1)
    return result

def CameraRotation(pixelincamera):
    rotation_matrix =[[np.cos(0.36),0,np.sin(0.36)],
                      [0, 1, 0],
                      [-np.sin(0.36), 0, np.cos(0.36)]]
    return np.dot(rotation_matrix, pixelincamera)

def pixelBody2World(opticalinbody,theta): 
    Transform_matrix = [[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]
    result = np.dot(Transform_matrix,opticalinbody)
    return result


