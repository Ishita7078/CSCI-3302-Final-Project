"""grocery controller."""

# Apr 1, 2025

from controller import Robot
import math
import numpy as np
from iterative_closest_point import icp_matching
import matplotlib.pyplot as plt


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
# lidar_offsets = lidar_offsets[::-1]

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

SCALE = 20

map_size = (30, 16)  # meters
resolution = 0.0033  # meters per cell
grid_width = int(map_size[0] * SCALE)
grid_height = int(map_size[1] * SCALE)
occupancy_grid = np.zeros((grid_width, grid_height))
# occupancy_grid = np.zeros((300, 300))
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
# lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] #provides clearest image

def get_pose(gps, compass): #webots provided pose, CHANGE
    x_r = gps.getValues()[0]
    y_r = gps.getValues()[1]
    theta_r = np.arctan2(compass.getValues()[0], compass.getValues()[1])
    return x_r, y_r, theta_r

def rotate(ranges, angles): #rotate angles appropriately
    xs = ranges * np.cos(angles)
    ys = ranges * np.sin(angles)
    return np.vstack((xs, ys))


def to_pixels(x, y):
   return int((x + 15) * SCALE), int((y + 8.05) * SCALE)
    
  
def to_world(bin, distance):
    global pose_x, pose_y, pose_theta  
    if distance != float('inf') and distance > 0 and distance < 5:
        print(distance)
        
        x_robot = distance * np.cos(lidar_offsets[bin])
        y_robot = distance * np.sin(lidar_offsets[bin])

        transform = np.array([ #homogenous transformation matrix
            [np.cos(pose_theta), -np.sin(pose_theta), pose_x],
            [np.sin(pose_theta), np.cos(pose_theta), pose_y],
            [0, 0, 1]
        ])
        
        # robot = np.array([pose_x, pose_y, 1])
        robot = np.array([x_robot, y_robot, 1])

        world = np.dot(transform, robot)

        return world[0], world[1] 
    return None


def line_algo(x0, y0, x1, y1): #bresenham's line algorithm
    # print(x0, y0, x1, y1)
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x_inc = 1 if x0 < x1 else -1 #takes care of lines going either way
    s_inc = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if occupancy_grid[x0][y0] != 2 and occupancy_grid[x0][y0] != 3: #make sure pixel is not an obstacle or on the robot path before filling it in with white
            occupancy_grid[x0][y0] = 1 
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += x_inc
        if e2 < dx:
            err += dx
            y0 += s_inc

# ------------------------------------------------------------------
# Helper Functions

poses = []  #list of (x, y, theta)
scans = []  #list of LIDAR
last_scan = None
last_pose = None

gripper_status="closed"

# Main Loop
while robot.step(timestep) != -1:
    print("Lidar max range:", lidar.getMaxRange())

        
    pose_x, pose_y, pose_theta = get_pose(gps, compass) #webots pose, CHANGE
    # print(pose_x, pose_y, pose_theta)
    lidar_values = np.array(lidar.getRangeImage())

    # display.setColor(0xFF0000)  # Green line

    # Draw the line (x0, y0) to (x1, y1)
    # display.drawPixel(400, 300)
    
    # scan = lidar_valxues[83:-83]  #clearest image

    # valid = np.isfinite(lidar_values) & (lidar_values > 3) #filter invalid readings

    # filtered_ranges = lidar_values[valid]
    # filtered_angles = lidar_offsets[valid]   
    # # print(lidar_values[0])
    
    # if len(filtered_ranges) > 10:
    #     curr = rotate(filtered_ranges, filtered_angles)
    
    #     if 'pc_last' in locals(): #if previous scan, apply icp
    #         try:
    #             R, t = icp_matching(pc_last, curr)
    #             dtheta = math.atan2(R[1, 0], R[0, 0])
    #             dx, dy = t[0, 0], t[1, 0]
    
    #             pose_theta += dtheta
    #             pose_x += dx * math.cos(pose_theta) - dy * math.sin(pose_theta)
    #             pose_y += dx * math.sin(pose_theta) + dy * math.cos(pose_theta)
    
    #         except Exception as e: #to catch improperly formatted dimensions or non-convergence of SVD
    #             print("[ICP Error]", e)
    
    #     pc_last = curr
    
    # #update last scan and pose
    # last_scan = lidar_values
    # last_pose = (pose_x, pose_y, pose_theta)

    pose_pixels = to_pixels(pose_x, pose_y)
    # print(pose_pixels[0], pose_pixels[1])
    # display.setColor(0xFF0000)
    # display.drawPixel(pose_pixels[0], pose_pixels[1])
    occupancy_grid[pose_pixels[0]][pose_pixels[1]] = 2
    # print(occupancy_grid[pose_pixels[0]][pose_pixels[1]])

    robot_x, robot_y = to_pixels(pose_x, pose_y)

    for i in range(len(lidar_offsets)):
        distance = lidar_values[i]
        world_coords = to_world(i, lidar_values[i])
        if world_coords is not None: #if distance is infinity world coords will be none
            # print("HIHIHI")
            world_x, world_y = to_pixels(world_coords[0], world_coords[1])

    
            if 0 <= world_x < grid_width and 0 <= world_y < grid_height:
                line_algo(robot_x, robot_y, world_x, world_y)
                # print(f"[DEBUG] Marking map at ({world_x}, {world_y})")
                # print(occupancy_grid[world_x][world_y])
                if distance < 1 and distance > 0.3:
                    occupancy_grid[world_x][world_y] = 3  


    ##### Part 4: Draw the obstacle and free space pixels on the map
    # for i, row in enumerate(occupancy_grid): #loop through 2x2 array representing the world and set display colors(sorry this makes it so slow)
    #     for j, cell in enumerate(row):
    #         if cell == 0: #black
    #             display.setColor(0x000000)
    #         elif cell == 1: #white
    #             display.setColor(0xFFFFFF)
    #         elif cell == 2: #red
    #             print("AA")
    #             display.setColor(0xFF0000)
    #         elif cell == 3: #blue
    #             display.setColor(0x0000FF)
            
    #         display.drawPixel(i, j)
    color_grid = np.zeros((grid_width, grid_height, 3), dtype=np.uint8)
    # print(color_grid)

    color_grid[occupancy_grid == 1] = [255, 255, 255]  # Free space = White
    color_grid[occupancy_grid == 2] = [255, 0, 0]      # Robot path = Red
    color_grid[occupancy_grid == 3] = [0, 0, 255]      # Obstacles = Blue
    # Unknown space remains black (0,0,0)
    # print(color_grid[975][4499])
    # print(color_grid[975-10:975+10, 4499-10:4499+10])
    plt.clf()
    plt.imshow(color_grid, origin="lower")
    plt.title("Occupancy Grid Map (30x16m)")
    plt.draw()  # Update the plot
    plt.pause(0.01)  # Keep the plot updated  


    # plt.clf()
    # plt.imshow(occupancy_grid, cmap="gray", origin="lower")
    # plt.title("Occupancy Grid Map (30x16m)")
    # plt.pause(0.01)
    
    
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
