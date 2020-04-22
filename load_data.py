dataset = 20
import numpy as np
class encoder:
    def __init__(self):
        with np.load("/data/Encoders%d.npz"%dataset) as data:
            self.encoder_counts = data["counts"] # 4 x n encoder counts
            self.encoder_stamps = data["time_stamps"] # encoder time stamps

class lidar:
    def __init__(self):
        with np.load("/data/Hokuyo%d.npz"%dataset) as data:
            self.lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
            self.lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
            self.lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
            self.lidar_range_min = data["range_min"] # minimum range value [m]
            self.lidar_range_max = data["range_max"] # maximum range value [m]
            self.lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
            self.lidar_stamsp = data["time_stamps"]  # acquisition times of the lidar scans

class imu:
    def __init__(self):
        with np.load("/data/Imu%d.npz"%dataset) as data:
            self.imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
            self.imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
            self.imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements

class kinect:
    def __init__(self):
        with np.load("/data/Kinect%d.npz"%dataset) as data:
            self.disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
            self.rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images
