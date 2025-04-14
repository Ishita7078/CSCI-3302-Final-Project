"""grocery controller."""

# Apr 1, 2025

from controller import Robot
import math
import numpy as np
from iterative_closest_point import icp_matching

#Initialization
print("=== Initializing Grocery Shopper...")
#Consts
MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)
print(LIDAR_ANGLE_BINS)
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE / 2., LIDAR_ANGLE_RANGE / 2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[::-1]

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())



# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")

# 

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.35, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.045,0.045)

robot_parts={}
for i, part_name in enumerate(part_names):
    robot_parts[part_name]=robot.getDevice(part_name)
    robot_parts[part_name].setPosition(float(target_pos[i]))
    robot_parts[part_name].setVelocity(robot_parts[part_name].getMaxVelocity() / 2.0)

# Enable gripper encoders (position sensors)
left_gripper_enc=robot.getDevice("gripper_left_finger_joint_sensor")
right_gripper_enc=robot.getDevice("gripper_right_finger_joint_sensor")
left_gripper_enc.enable(timestep)
right_gripper_enc.enable(timestep)

# Enable Camera
camera = robot.getDevice('camera')
camera.enable(timestep)
# camera.recognitionEnable(timestep)

# Enable GPS and compass localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# Enable LiDAR
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# Enable display
display = robot.getDevice("display")

keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] #provides clearest image

def get_pose(gps, compass): #webots provided pose, CHANGE
    x_r = gps.getValues()[0]
    y_r = gps.getValues()[1]
    theta_r = np.arctan2(compass.getValues()[0], compass.getValues()[1])
    return x_r, y_r, theta_r

def rotate(ranges, angles): #rotate angles appropriately
    xs = ranges * np.cos(angles)
    ys = ranges * np.sin(angles)
    return np.vstack((xs, ys))

# ------------------------------------------------------------------
# Helper Functions

poses = []  #list of (x, y, theta)
scans = []  #list of LIDAR
last_scan = None
last_pose = None

gripper_status="closed"

# Main Loop
while robot.step(timestep) != -1:
        
    pose_x, pose_y, pose_theta = get_pose(gps, compass) #webots pose, CHANGE
    lidar_values = np.array(lidar.getRangeImage())
    
    scan = lidar_values[83:-83]  #clearest image

    valid = np.isfinite(scan) #filter invalid readings
    filtered_ranges = scan[valid]
    filtered_angles = lidar_offsets[valid]    
    
    if len(filtered_ranges) > 10:
        curr = rotate(filtered_ranges, filtered_angles)
    
        if 'pc_last' in locals(): #if previous scan, apply icp
            try:
                R, t = icp_matching(pc_last, curr)
                dtheta = math.atan2(R[1, 0], R[0, 0])
                dx, dy = t[0, 0], t[1, 0]
    
                pose_theta += dtheta
                pose_x += dx * math.cos(pose_theta) - dy * math.sin(pose_theta)
                pose_y += dx * math.sin(pose_theta) + dy * math.cos(pose_theta)
    
            except Exception as e: #to catch improperly formatted dimensions or non-convergence of SVD
                print("[ICP Error]", e)
    
        pc_last = curr
    
    #update last scan and pose
    last_scan = scan
    last_pose = (pose_x, pose_y, pose_theta)
    
    
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
    
    if(gripper_status=="open"):
        # Close gripper, note that this takes multiple time steps...
        robot_parts["gripper_left_finger_joint"].setPosition(0)
        robot_parts["gripper_right_finger_joint"].setPosition(0)
        if right_gripper_enc.getValue()<=0.005:
            gripper_status="closed"
    else:
        # Open gripper
        robot_parts["gripper_left_finger_joint"].setPosition(0.045)
        robot_parts["gripper_right_finger_joint"].setPosition(0.045)
        if left_gripper_enc.getValue()>=0.044:
            gripper_status="open"
