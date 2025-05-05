"""grocery controller."""

# Apr 1, 2025

from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import cv2
from scipy.signal import convolve2d
import random
from ultralytics import YOLO
import builtins
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
from ikpy.utils import geometry

from controller import Supervisor # TODO: remove

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
# print(LIDAR_ANGLE_BINS)
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE / 2., LIDAR_ANGLE_RANGE / 2., LIDAR_ANGLE_BINS)
# lidar_offsets = lidar_offsets[::-1]

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")

base_elements=["base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint", "torso_lift_link", "torso_lift_link_TIAGo front arm_11367_joint", "TIAGo front arm_11367"]
my_chain = Chain.from_urdf_file("robot_urdf.urdf", base_elements=["base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint", "torso_lift_link", "torso_lift_link_TIAGo front arm_11367_joint", "TIAGo front arm_11367"])

my_chain.active_links_mask = [False] * len(my_chain.links)

for link_id in builtins.range(len(my_chain.links)):
    # This is the actual link object
    link = my_chain.links[link_id]

    # I've disabled "torso_lift_joint" manually as it can cause
    # the TIAGO to become unstable.
    if hasattr(link, 'joint_type') and (link.joint_type == 'revolute' or link.name == "gripper_right_finger_joint"):
        my_chain.active_links_mask[link_id] = True
    else:
    # if link.name not in part_names or link.name == "torso_lift_joint":
        print("Disabling {}".format(link.name))
        my_chain.active_links_mask[link_id] = False
#
# print(my_chain.links)
# print(my_chain.active_links_mask)

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.09, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf')
robot_parts=[]

for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)

# Enable Camera
camera = robot.getDevice('camera')
camera.enable(timestep)
# camera.recognitionEnable(timestep)
width = camera.getWidth()
height = camera.getHeight()

# Load your trained YOLOv5 model
model = YOLO("best.pt")
model.conf = 0.5  # confidence threshold
model.iou = 0.4   # IOU threshold for NMS

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

# Get the position sensor for the left wheel
left_wheel_sensor = robot.getDevice('wheel_left_joint_sensor')
# Enable the sensor with a sampling period of 'timestep'
left_wheel_sensor.enable(timestep)

# Get the position sensor for the left wheel
right_wheel_sensor = robot.getDevice('wheel_right_joint_sensor')
# Enable the sensor with a sampling period of 'timestep'
right_wheel_sensor.enable(timestep)

prev_left_position = left_wheel_sensor.getValue()
prev_right_position = right_wheel_sensor.getValue()

keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# Odometry
pose_x     = -5
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

# MODE = 'map'
# MODE = 'planner'
MODE = 'navigation'

vrb = False # Verbose print in joint helper functions

# Initialize the arm motors and encoders.
motors = []
for link in my_chain.links:
    if link.name in part_names and link.name != "torso_lift_joint":
        motor = robot.getDevice(link.name)

        # Make sure to account for any motors that
        # require a different maximum velocity!
        if link.name == "torso_lift_joint":
            motor.setVelocity(0.07)
        else:
            motor.setVelocity(1)

        position_sensor = motor.getPositionSensor()
        position_sensor.enable(timestep)
        motors.append(motor)


# Pixel Map Dimension
HEIGHT = 914
WIDTH = 494
SCALE = 30

map_size = (30, 16)  # meters
resolution = 0.0033  # meters per cell
grid_width = int(map_size[0] * SCALE)
grid_height = int(map_size[1] * SCALE)
occupancy_grid = np.zeros((grid_width, grid_height))
# occupancy_grid = np.zeros((300, 300))
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)

wait_timer = 0
def stalled_for(timesteps):
    # Stall the main loop for a set amount of timesteps
    global wait_timer
    if wait_timer > timesteps:
        # Once enough time has stalled, reset and return True
        wait_timer = 0
        return True
    else:
        wait_timer += 1
        return False

def filter(waypoints, min_distance):
    # Filters a list of waypoints to be min_distance apart
    filtered = [waypoints[0]]
    for i in builtins.range(1, len(waypoints)):
        distance = position_error(filtered[-1][0], filtered[-1][1], waypoints[i][0], waypoints[i][1])
        if distance >= min_distance:
            filtered.append(waypoints[i])
    return filtered

####
# Inverse Kinematics Helper Functions
####

def position_error(current_x, current_y, goal_x, goal_y):
    # Distance between current position and goal
    return math.sqrt((goal_x - current_x)**2 + (goal_y - current_y)**2)


def bearing_error(current_x, current_y, current_theta, goal_x, goal_y):
    # Direction from current position to goal
    y = goal_y - current_y
    x = goal_x - current_x
    alpha = math.atan2(y, x)
    alpha -= current_theta
    alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
    return alpha + math.pi/2


def heading_error(current_theta, goal_theta):
    # Angle difference between current theta and goal theta
    eta = goal_theta - current_theta
    eta = (eta + math.pi) % (2 * math.pi) - math.pi
    return eta + math.pi/2

# Inverse Kinematics Controller
def ik_controller(vL, vR, x_i, y_i, pose_x, pose_y, pose_theta, waypoints, index):
    # print(pose_x, pose_y, pose_theta)

    # STEP 1: Calculate the error
    pose_theta += math.pi * 0.5
    rho = position_error(pose_x, pose_y, waypoints[index][0], waypoints[index][1])
    alpha = bearing_error(pose_x, pose_y, pose_theta, waypoints[index][0], waypoints[index][1])
    theta_g = math.atan2(waypoints[index][1] - y_i, waypoints[index][0] - x_i)
    eta = heading_error(pose_theta, theta_g)
    # print(f"rho: {rho}, alpha: {alpha}, eta: {eta}")
    # STEP 2: Controller
    velocity = 0.09 * rho
    angular = 0.06 * alpha + 0.03 * eta
    # Check if we are close to the goal
    if rho < 0.2:
        x_i = waypoints[index][0]
        y_i = waypoints[index][1]
        index += 1
    else:
        # Set wheel speeds
        vL = (velocity - (angular * AXLE_LENGTH * 0.5)) / (0.02)
        vR = (velocity + (angular * AXLE_LENGTH * 0.5)) / (0.02)
        lsign = 1 if vL > 0 else -1
        rsign = 1 if vR > 0 else -1
        ratio = abs(vL / vR)
        # Limit speeds
        if abs(vL) > MAX_SPEED or abs(vR) > MAX_SPEED:
            if ratio > 1:
                vL = MAX_SPEED * lsign
                vR = MAX_SPEED / ratio * rsign
            else:
                vL = MAX_SPEED * ratio * lsign
                vR = MAX_SPEED * rsign
        if abs(abs(vL / vR) - ratio) > 1e-10: exit()
        # Turn sharp
        if alpha < -0.09:
            vL = (2)
            vR = (-2)
        elif alpha > 0.09:
            vL = (-2)
            vR = (2)

    # Adjust wheel speed range
    if vL != 2 and vL != -2: vL = (vL / MAX_SPEED) * 2
    if vR != 2 and vR != -2: vR = (vR / MAX_SPEED) * 2

    # print(f"vL: {vL}, vR: {vR}")
    return vL, vR, x_i, y_i, index


# Get pose from gps and compass
def get_pose(gps, compass):
    n = compass.getValues()
    x = gps.getValues()[0]-0.2*n[1] # Adjust pose for difference between GPS and true pose
    y = gps.getValues()[1]-0.2*n[0] 

    theta = math.atan2(n[0], n[1])
    return x,y,theta

reset_tracker = 0
# Calculate odometry using change in wheel position
def odometry():
    global vL, vR, pose_x, pose_y, pose_theta, prev_left_position, prev_right_position, left_wheel_sensor, right_wheel_sensor, gps, compass, reset_tracker
    radius = .0982 #wheel radius

    #odometry
    if reset_tracker < 32:
        position_left = left_wheel_sensor.getValue()
        position_right = right_wheel_sensor.getValue()

        #change in wheel positions from previous timestep to now
        change_in_left_wheel = (position_left - prev_left_position)
        change_in_right_wheel = (position_right - prev_right_position)

        #linear distance traveled by each wheel
        dist_left = change_in_left_wheel * radius
        dist_right = change_in_right_wheel * radius

        #distance traveled by both wheels and theta
        dist = (dist_right + dist_left) * 0.5
        theta = (dist_right - dist_left) / AXLE_LENGTH

        pose_x += dist * math.cos(pose_theta)
        pose_y += dist * math.sin(pose_theta)

        pose_theta += theta
        pose_theta = (pose_theta + math.pi) % (2 * math.pi) - math.pi

        #current wheel positions become prev for next loop
        prev_left_position = position_left
        prev_right_position = position_right
        reset_tracker +=1
    #correction with gps every 32 timesteps
    else:
        pose_x, pose_y, pose_theta = get_pose(gps,compass)
        reset_tracker = 0
    return pose_x, pose_y, pose_theta


# Convert from world coords to pixels
def to_pixels(x,y):
    x = int((x + map_size[0]/2) / map_size[0] * HEIGHT)
    y = int((y + map_size[1]/2) / map_size[1] * WIDTH + 10)
    return x,y

# Convert from pixels to world coords
def from_pixels(x, y):
    x = (x / HEIGHT) * map_size[0] - map_size[0] / 2
    y = (y / WIDTH) * map_size[1] - map_size[1] / 2
    return x, y

def to_world(bin, distance):
    global pose_x, pose_y, pose_theta  
    if distance != float('inf') and distance > 0 and distance < 5:
        # print(distance)
        
        x_robot = distance * np.cos(lidar_offsets[bin])
        y_robot = distance * np.sin(lidar_offsets[bin])

        transform = np.array([ #homogenous transformation matrix
            [np.cos(pose_theta), np.sin(pose_theta), pose_x],
            [-np.sin(pose_theta), np.cos(pose_theta), pose_y],
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

#####
# RRT Star Helper Functions
#####
class Node:
    def __init__(self, pt, parent=None):
        self.point = pt  # n-Dimensional point
        self.parent = parent  # Parent node
        self.path_from_parent = []  # List of points along the way from the parent node (for visualization)


def get_distance_helper(point1,point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

#returns node from node_list that is closest to the target node
def get_nearest_vertex(node_list, q_point):
    #loop through list and find closest node
    closest_vertex = None
    for node in node_list:
        if closest_vertex==None:
            closest_vertex = node
        else:
            node_distance = get_distance_helper(q_point, node.point)
            current_closest_distance = get_distance_helper(q_point, closest_vertex.point)
            if node_distance < current_closest_distance:
                closest_vertex = node
    return closest_vertex


def steer(from_point, to_point, delta_q):
    path = []
    #get distance between points and see if it is in range
    distance = get_distance_helper(from_point,to_point)
    #if distance is too big get a new point along the line
    if(distance > delta_q):
        direction = (to_point-from_point)/distance
        new_to_point = delta_q*direction+from_point
        line = np.linspace(from_point, new_to_point, num=10)
    else:
        line = np.linspace(from_point, to_point, num=10)

    # Convert to list of tuples of ints
    return [np.array((int(p[0]), int(p[1]))) for p in line]

#generate random new node
def get_random_valid_vertex(state_valid, convolved_map):
    vertex = None
    while vertex is None: # Get starting vertex
        random_x = np.random.randint(0,WIDTH)
        random_y = np.random.randint(0,HEIGHT)
        pt = np.array((random_x,random_y))
        #check if point is not marked as a 1 on the map
        if state_valid(pt, convolved_map):
            vertex = pt
    return vertex

#get list of nodes near a point
def near(node_list,q_new,r):
    near_list = []
    for node in node_list:
        distance = get_distance_helper(node.point,q_new)
        if(distance <=r):
            near_list.append(node)
    return near_list

#check if the given point is free or is in the way of an obstacle 
def state_is_valid(pt,convolved_map):
    x = int(pt[0])
    y = int(pt[1])
    if 0 <= x < WIDTH:
        if 0 <= y < HEIGHT:
            if(convolved_map[y,x] == 0):
                return True
    return False

def rrt_star(convolved_map, state_is_valid, starting_point, goal_point, k, delta_q):
    #   RRT* Pseudo Code CREDIT: https://www.ri.cmu.edu/pub_files/2014/9/TR-2013-JDG003.pdf
    node_list = []
    cost_list = {}
    first = Node(starting_point, parent=None)
    node_list.append(first)  # Add Node at starting point with no parent
    cost_list.update({tuple(first.point): 0})
    rad = delta_q * 1.5
    for i in range(1, k):
        #condition to randomly pick the new node to be the goal_point occasionally
        if goal_point is not None and random.random() < 0.05:
            q_rand = goal_point
        else:
            q_rand = get_random_valid_vertex(state_is_valid, convolved_map) #get new node
        q_nearest = get_nearest_vertex(node_list, q_rand)
        path_rand_nearest = steer(q_nearest.point, q_rand, delta_q) #get the path between nodes
        q_new = path_rand_nearest[-1] #in case the new node was replaced by a closer node we set q_new to the last node in the line
        valid = True
        #validity check
        for point in path_rand_nearest:
            if state_is_valid(point, convolved_map) == False:
                valid = False
        if valid:
            #create new node
            new_node = Node(q_new, q_nearest)
            new_node.path_from_parent = path_rand_nearest
            node_list.append(new_node)
            #calculate cost of node
            node_cost = cost_list[tuple(new_node.parent.point)] + get_distance_helper(new_node.point, new_node.parent.point)
            cost_list.update({tuple(new_node.point): node_cost})
            q_near_list = near(node_list, new_node.point, rad) # find neighbors for rewiring

            #check for best parent node
            q_min = q_nearest
            c_min = cost_list.get(tuple(q_nearest.point)) + get_distance_helper(q_nearest.point, new_node.point)

            for node in q_near_list:
                near_node_cost = cost_list[tuple(node.point)] + get_distance_helper(node.point, new_node.point)
                if (near_node_cost < c_min):
                    node_path = np.linspace(node.point, new_node.point, num=10)
                    path_valid = True
                    for point in node_path:
                        if state_is_valid(point, convolved_map) == False:
                            path_valid = False
                    if (path_valid):
                        q_min = node
                        c_min = near_node_cost

            # reassign best parent and update cost   
            new_node.parent = q_min
            new_node.path_from_parent = np.linspace(new_node.parent.point, new_node.point, num=10)
            cost_list[tuple(new_node.point)] = c_min

            #rewire neighbors
            for node in q_near_list:
                c_near = cost_list[tuple(node.point)]
                c_new = cost_list[(tuple(new_node.point))] + get_distance_helper(node.point, new_node.point)
                if (c_new < c_near):
                    node_parent_path = np.linspace(new_node.point, node.point, num=10)
                    path_valid = True
                    for point in node_parent_path:
                        if state_is_valid(point, convolved_map) == False:
                            path_valid = False
                    if (path_valid):
                        node.parent = new_node
                        node.path_from_parent = np.linspace(new_node.point, node.point, num=10)
                        cost_list[tuple(node.point)] = c_new

            #found goal
            if (goal_point is not None):
                distance_from_goal = get_distance_helper(new_node.point, goal_point)
                if (distance_from_goal < 1e-5):
                    print("Found the end!")
                    return node_list

    print(f"Did not find end, finished after {k} iterations.")
    print(f"Pixels: {(starting_point)}:{convolved_map[starting_point[1]][starting_point[0]]}, {(goal_point)}:{convolved_map[goal_point[1]][goal_point[0]]}")
    print(f"World: {from_pixels(starting_point)}, {from_pixels(goal_point)}")
    return node_list

#For printing RRT* Map
def visualize_2D_graph(convolved_map, nodes, goal_point=None, filename=None):
        fig = plt.figure()
        plt.xlim(0,WIDTH)
        plt.ylim(0,HEIGHT)

        goal_node = None
        for node in nodes:
            if node.parent is not None:
                node_path = np.array(node.path_from_parent)
                plt.plot(node_path[:,0], node_path[:,1], '-b')
            if goal_point is not None and np.linalg.norm(node.point - np.array(goal_point)) <= 1e-5:
                goal_node = node
                plt.plot(node.point[0], node.point[1], 'k^')
            else:
                plt.plot(node.point[0], node.point[1], 'ro')

        plt.plot(nodes[0].point[0], nodes[0].point[1], 'ko')

        if goal_node is not None:
            cur_node = goal_node
            while cur_node is not None:
                if cur_node.parent is not None:
                    node_path = np.array(cur_node.path_from_parent)
                    plt.plot(node_path[:,0], node_path[:,1], '--y')
                    cur_node = cur_node.parent
                else:
                    break

        if goal_point is not None:
            plt.plot(goal_point[0], goal_point[1], 'gx')


        if filename is not None:
            fig.savefig(filename)
        else:
            plt.show()

# ------------------------------------------------------------------
# Joint Helper Functions
# ------------------------------------------------------------------
def moveArmToTarget(ikResults):
    # '''Moves arm given ikResults'''
    # Set the robot motors
    for res in builtins.range(len(ikResults)):
        if my_chain.links[res].name in part_names:
            # This code was used to wait for the trunk, but now unnecessary.
            # if abs(initial_position[2]-ikResults[2]) < 0.1 or res == 2:
            robot.getDevice(my_chain.links[res].name).setPosition(ikResults[res])
            if vrb:
                print("Setting {} to {}".format(my_chain.links[res].name, ikResults[res]))


def calculateIk(target_position,  orient=True, orientation_mode="Y", target_orientation=[0,0,-0.5]):
    '''Calculate inverse kinematics of robot arm given end effector pose in robot coordinates'''
    initial_guess = []
    # Set all links to be within their bounds
    for i, link in enumerate(my_chain.links):
        if my_chain.active_links_mask[i]:
            lower, upper = link.bounds
            if lower != -np.inf and upper != np.inf:
                initial_guess.append((lower + upper) / 2.0)
            else:
                initial_guess.append(0.0)
        else:
            if link.name == 'torso_lift_joint':
                # Set torso lift joint to constant height
                initial_guess.append(0.2)
                # print(f"Setting {link.name}")
            else:
                initial_guess.append(0.0)
    initial_guess = np.array(initial_guess)
    # Calculate IK
    ik_result = my_chain.inverse_kinematics(
        target_position=np.array(target_position),
        target_orientation=target_orientation,
        orientation_mode=orientation_mode,
        initial_position=initial_guess
    )
    return ik_result


# Close the robot gripper
def closeGrip():
    robot.getDevice("gripper_right_finger_joint").setPosition(0.0)
    robot.getDevice("gripper_left_finger_joint").setPosition(0.0)


# Open the robot gripper
def openGrip():
    robot.getDevice("gripper_right_finger_joint").setPosition(0.045)
    robot.getDevice("gripper_left_finger_joint").setPosition(0.045)


# Planner mode to create convolved map and test RRT*
if MODE == "planner":
    #declare start and end
    start = np.array((20,20))
    end = np.array((312,807))
    # Load map
    map = np.load("map.npy")
    map = map==3

    #Convolve map
    KERNEL_DIM = 40
    kernel = np.ones(shape=[KERNEL_DIM, KERNEL_DIM])
    convolved_map = convolve2d(map, kernel)
    convolved_map = convolved_map > 0
    convolved_map = convolved_map * 1
    height,width = convolved_map.shape
    np.save("convolved_map.npy", convolved_map)

    #call path planner
    waypoints_all = rrt_star(convolved_map, state_is_valid, start, end, 2000, 30)

    plt.imshow(convolved_map)
    #UNCOMMENT TO SEE ALL POINTS IN RRT*
    # for waypt in waypoints_all:
    #     #plt.plot(waypt.point[0], waypt.point[1], marker='o', color='blue', markersize=2)

    #uncomment lines below to see all trees
    goal_node = None
    for node in waypoints_all:
        if node.parent is not None:
            node_path = np.array(node.path_from_parent)
            #plt.plot(node_path[:,0], node_path[:,1], '-b')
        if np.linalg.norm(node.point - np.array(end)) <= 1e-5:
            goal_node = node
            #plt.plot(node.point[0], node.point[1], 'k^')
        # else:
        #     plt.plot(node.point[0], node.point[1], 'ro')

    plt.plot(waypoints_all[0].point[0], waypoints_all[0].point[1], 'ko')

    #waypoints for the path!
    path_waypoints = []

    if goal_node is not None:
        cur_node = goal_node
        while cur_node is not None:
            if cur_node.parent is not None:
                plt.plot(cur_node.point[0], cur_node.point[1], marker='o', color='blue', markersize=2)
                path_waypoints.append(cur_node.point)
                node_path = np.array(cur_node.path_from_parent)
                plt.plot(node_path[:,0], node_path[:,1], '--y')
                cur_node = cur_node.parent
            else:
                break

    if goal_node is not None:
        plt.plot(goal_node.point[0], goal_node.point[1], 'gx')

    # print(path_waypoints)

    path_world_coords = []
    for point in path_waypoints:
        path_world_coords.append(((point[1]/30) - 15, (point[0] /30)-8.05))
    np.save("path.npy", path_world_coords)

    # print(path_world_coords)

    path_map = np.load("path.npy")

    # x_coords, y_coords = zip(*waypoints_points)
    # plt.plot(x_coords, y_coords, color='red', linewidth=1)
    plt.plot(start[0], start[1], marker='^', color='lightgreen', markersize=7)
    plt.plot(end[0], end[1], marker='^', color='red', markersize=7)
    plt.show()
    visualize_2D_graph(convolved_map,waypoints_all,end,"test2.png")
# ------------------------------------------------------------------

# Tiago Starting position
y_i = 0
x_i = -5

# Path through grocery store to follow
aisle_path = [(-4.83, 5.82),(3.41, 5.82),(13.15, 5.82),(13.15,2.18),(3.41,2.18),(-4.83,2.18),(-4.83,-1.82),(3.41,-1.82),(13.15,-1.82),(13.15,-5.91),(3.41,-5.91),(-4.83,-5.91)]
aisle_state = -1 # state in the aisle path
current_path = [] # current path between points on the aisle path
state = 0 # state in the current path

ARM_STATE = 0 # Arm controller state machine
arm_path = [] # Path for arm to follow
arm_index = 0 # Index along the arm_path

TASK = "follow_aisle" # Current overall task
detection_timer = 0 # Frame counter between calling ML cube detection
timesteps_without_detection = 0 # Timesteps since last cube detection
WALL_MODE = "not_done" # Wall following mode
cube_bounds = None # Bounds of detected cube

UPPER_SHELF = 1.12 # Upper shelf height
LOWER_SHELF = 0.7 # Lower shelf height
min_lidar = 10 # Min lidar value

cmap = np.load("convolved_map.npy")
# print(f"shape: {cmap.shape}")
# plt.imshow(cmap)
# plt.show()

armTop = (0,0,2) # Position arm above head
armTopIk = calculateIk(armTop)

# Main Loop
while robot.step(timestep) != -1 and MODE != 'planner':
    pose_x, pose_y, pose_theta = odometry() # Get pose from odometry
    if np.isnan(pose_x) or np.isnan(pose_y) or np.isnan(pose_theta):
        # Odometry values are occasionally initialized to NaN
        pose_x, pose_y, pose_theta = -5,0,0
        print("NaN pose at first iteration!")
        continue

    # Get lidar values
    lidar_values = np.array(lidar.getRangeImage())

    # Mapping mode
    if MODE == 'map':

        # print(pose_x, pose_y, pose_theta)
        pose_pixels = to_pixels(pose_x, pose_y)
        occupancy_grid[pose_pixels[0]][pose_pixels[1]] = 2
        robot_x, robot_y = to_pixels(pose_x, pose_y)



        for i in range(len(lidar_offsets)):
            distance = lidar_values[i]
            world_coords = to_world(i, lidar_values[i])
            if world_coords is not None: #if distance is infinity world coords will be none
                world_x, world_y = to_pixels(world_coords[0], world_coords[1])


                if 0 <= world_x < grid_width and 0 <= world_y < grid_height:
                    line_algo(robot_x, robot_y, world_x, world_y)
                    if distance < 5 and distance > 0:
                        occupancy_grid[world_x][world_y] = 3
        color_grid = np.zeros((grid_width, grid_height, 3), dtype=np.uint8)

        color_grid[occupancy_grid == 1] = [255, 255, 255]  #free space = White
        color_grid[occupancy_grid == 2] = [255, 0, 0]      #robot path = Red
        color_grid[occupancy_grid == 3] = [0, 0, 255]      #obstacles = Blue
        plt.clf()
        plt.imshow(color_grid, origin="lower")
        plt.draw()
        plt.pause(0.01)

        center_bin = LIDAR_ANGLE_BINS // 2
        left_bin = int(center_bin - 100)
        right_bin = int(center_bin + 100)

        # # Thresholds for "clear"
        forward_clear = lidar_values[center_bin] > 1.0

        if forward_clear:
            vL, vR = 10, 10  # move forward
        else:
            vL, vR = 1, -1  # if boxed in, turn right


        robot_parts[MOTOR_LEFT].setVelocity(vL)
        robot_parts[MOTOR_RIGHT].setVelocity(vR)

        file_path = os.path.join(os.getcwd(), "occupancy_grid.npy")
        np.save(file_path, occupancy_grid)
        print(f"Occupancy grid saved to {file_path}")

    # Autonomous navigation mode
    # Follows aisle path through grocery store, and picks up each cube as it is encountered
    elif MODE == 'navigation':
        # Following wall and cube just went out of frame
        if WALL_MODE == "forward_left" or WALL_MODE == "forward_right":
            # Continue forward for x timesteps
            if WALL_MODE == "forward_left":
                vL, vR = 1.8, 2
            else:
                vL, vR = 2, 1.8
            # Move forward some distance based on distance to wall on the side
            if stalled_for(100*min_lidar):
                if WALL_MODE == "forward_left":
                    WALL_MODE = "turn_left"
                elif WALL_MODE == "forward_right":
                    WALL_MODE = "turn_right"

            robot_parts[MOTOR_LEFT].setVelocity(vL)
            robot_parts[MOTOR_RIGHT].setVelocity(vR)
            continue
        elif WALL_MODE == "turn_left":
            # After moving forward, turn left
            vL = -2
            vR = 2
            # Stall for length of time to turn 90 degrees
            if stalled_for(100):
                WALL_MODE = "done"
            robot_parts[MOTOR_LEFT].setVelocity(vL)
            robot_parts[MOTOR_RIGHT].setVelocity(vR)
            continue
        elif WALL_MODE == "turn_right":
            # After moving forward, turn right
            vL = 2
            vR = -2
            # Stall for length of time to turn 90 degrees
            if stalled_for(100):
                WALL_MODE = "done"
            robot_parts[MOTOR_LEFT].setVelocity(vL)
            robot_parts[MOTOR_RIGHT].setVelocity(vR)
            continue

        detections = []
        # Run ML cube detection only when needed
        if TASK != "grab_cube" and WALL_MODE not in ["forward_left", "forward_right", "turn_left", "turn_right"] and abs(pose_y - aisle_path[aisle_state][1]) < 1.5:
            if detection_timer > 10 or TASK == "go_to_cube": # Only check for detections every 10 timesteps, laggy
                detection_timer = 0
                img = camera.getImageArray()
                img_np = np.array(img, dtype=np.uint8).reshape((height, width, 3))  #3 for RGB channels

                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) #convert to BGR

                results = model(img_bgr)[0]  #inference
                detections = results.boxes.data.cpu().numpy()  #x1, y1, x2, y2, confidence, class
                class_names = model.names
            else:
                detection_timer += 1
        # Filter detections to be of length one
        if len(detections) > 1: detections = [detections[0]]
        filtered_detections = []
        green_count = 0
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            # size = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            label = class_names[int(cls)]
            if label == "green":
                green_count += 1
            # Only consider large square-ish cubes with a high degree of confidence
            if conf > 0.8 and (((x2-x1) > 4 and (y2-y1) > 4) or TASK == "go_to_cube"): # Confident it is a nearby cube
                if label == "yellow":
                    print(f"[DETECTION] {label} ({conf:.2f}) at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}], W {x2 - x1}, H {y2 - y1}")
                    filtered_detections.append(det)
        detections = filtered_detections

        # Very close cubes get detected as 3+ green cubes, so back up
        if green_count > 3:
            timesteps_without_detection = 0
            print("Too many greens, backing up!")
            vL, vR = -2,-2
            robot_parts[MOTOR_LEFT].setVelocity(vL)
            robot_parts[MOTOR_RIGHT].setVelocity(vR)
            continue

        # If there are no detections and the current task is to follow the aisle
        if detections == [] and TASK == "follow_aisle":
            # task = go to next point in aisle path, no rrt star since just straight lines with no obstacles?

            # If the end of the path has been reached
            if state < len(current_path):
                # Use IK controller to go to next waypoint
                vL, vR, x_i, y_i, state = ik_controller(vL, vR, x_i, y_i, pose_x, pose_y, pose_theta, current_path, state)
            else:
                # End of current path reached
                vL = 0
                vR = 0
                # Generate path to next point in the aisle path
                aisle_state += 1
                print(f"Generating new path from {(round(pose_x,2), round(pose_y,2))} to {aisle_path[aisle_state]}")
                start = to_pixels(pose_x, pose_y)
                start = np.array((start[1],start[0]))
                end = to_pixels(aisle_path[aisle_state][0], aisle_path[aisle_state][1])
                end = np.array((end[1],end[0]))
                # print(start, end)
                waypoints_all = rrt_star(cmap, state_is_valid, start, end, 5000, 5)
                plt.imshow(cmap)
                # uncomment lines below to see all trees
                goal_node = None
                for node in waypoints_all:
                    if node.parent is not None:
                        node_path = np.array(node.path_from_parent)
                        # plt.plot(node_path[:,0], node_path[:,1], '-b')
                    if np.linalg.norm(node.point - np.array(end)) <= 1e-5:
                        goal_node = node
                        # plt.plot(node.point[0], node.point[1], 'k^')
                    # else:
                    #     plt.plot(node.point[0], node.point[1], 'ro')

                plt.plot(waypoints_all[0].point[0], waypoints_all[0].point[1], 'ko')

                # waypoints for the path!
                path_waypoints = []

                if goal_node is not None:
                    cur_node = goal_node
                    while cur_node is not None:
                        if cur_node.parent is not None:
                            plt.plot(cur_node.point[0], cur_node.point[1], marker='o', color='blue', markersize=2)
                            path_waypoints.append(cur_node.point)
                            node_path = np.array(cur_node.path_from_parent)
                            plt.plot(node_path[:, 0], node_path[:, 1], '--y')
                            cur_node = cur_node.parent
                        else:
                            break

                if goal_node is not None:
                    plt.plot(goal_node.point[0], goal_node.point[1], 'gx')
                # print(path_waypoints)

                path_world_coords = []
                for point in reversed(path_waypoints):
                    path_world_coords.append(from_pixels(point[1], point[0]))
                current_path = path_world_coords
                current_path = filter(current_path, 1)
                current_path.append(path_world_coords[-1])
                current_path.pop(0)
                # print(current_path)
                # plt.show()
        else: # detections is not None or TASK == "grab_cube"
            if TASK != "grab_cube":
                # Move towards the cube using only camera and lidar values
                TASK = "go_to_cube"
                center_bin = LIDAR_ANGLE_BINS // 2
                print(f"Front lidar: {lidar_values[center_bin]}")
                # Find the largest detected cube
                largest_det = None
                largest_det_size = -1
                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    size = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if size > largest_det_size:
                        largest_det_size = size
                        largest_det = det
                # If there is a cube
                if largest_det is not None:
                    timesteps_without_detection = 0 # Reset detection timer
                    x1, y1, x2, y2, _, _ = largest_det

                    x_offset = (x1+x2)/2 - 120 # Calculate cube offset from center of camera
                    # Define turn speed magnitude based on how far from the center the cube is
                    turn_speed = x_offset/40
                    turn_speed = max(min(turn_speed,1),-1)

                    # Find minimum lidar value
                    min_lidar_index = 0
                    for i in builtins.range(len(lidar_values)):
                        if lidar_values[i] > 0.6 and lidar_values[i] < min_lidar and i < len(lidar_values):
                            min_lidar = lidar_values[i]
                            min_lidar_index = i
                    print(f"Min Lidar: {min_lidar} at {min_lidar_index}")
                    # If the minimum lidar value is on one of the sides, follow that wall
                    if (min_lidar_index > 380 or min_lidar_index < 280) and (WALL_MODE == "not_done"):
                        # Wall following
                        if min_lidar < 1.2:
                            n = compass.getValues()
                            theta = math.atan2(n[0], n[1])
                            if min_lidar_index > 380 and x_offset > -10:
                                # Wall on right
                                if math.pi/2 < theta < 3 or -math.pi/2 < theta < -0.15:
                                    # Spin to face parallel to wall
                                    vL = 0
                                    vR = 2.5
                                else:
                                    # Go straight along wall
                                    vL = 2
                                    vR = 2
                                    # When the cube is at the edge of the screen, stop using camera
                                    if x_offset > 100:
                                        WALL_MODE = "forward_right"
                                robot_parts[MOTOR_LEFT].setVelocity(vL)
                                robot_parts[MOTOR_RIGHT].setVelocity(vR)
                                continue
                            elif min_lidar_index < 280 and x_offset < 10:
                                # Wall on left
                                if 0.3 < theta < math.pi/2 or -math.pi/2 < theta < -0.3:
                                    # Spin to face parallel to wall
                                    vL = 2.5
                                    vR = 0
                                else:
                                    # Go straight along wall
                                    vL = 2
                                    vR = 2
                                    # When the cube is at the edge of the screen, stop using camera
                                    if x_offset < -100:
                                        WALL_MODE = "forward_left"
                                robot_parts[MOTOR_LEFT].setVelocity(vL)
                                robot_parts[MOTOR_RIGHT].setVelocity(vR)
                                continue

                    # Not following a wall, must be facing perpendicular to it

                    if lidar_values[center_bin] < 1.11:
                        # Too close to wall
                        vL = -1
                        vR = -1
                    elif 1.11 <= lidar_values[center_bin] <= 1.13 and abs(x_offset) < 3:
                        # Correct distance from wall and the cube is directly centered
                        # Stop at wall
                        vL = 0
                        vR = 0
                        # Reach for cube
                        TASK = "grab_cube"
                        cube_bounds = [x1, y1, x2, y2]
                    else:
                        # Turn to face the cube
                        VMAX = 2 # Maximum wheel motor speed
                        # Calculate speed based on front lidar distance and turn speed from above
                        speed = max(1, min(VMAX, (lidar_values[center_bin] - 1.13) * 5 + 1))
                        vL = speed * (1+turn_speed)
                        vR = speed * (1-turn_speed)
                        # Tight turn
                        if abs(vL) < 1 and abs(vR) > VMAX/2:
                            vL = -abs(vR)
                        elif abs(vR) < 1 and abs(vL) > VMAX/2:
                            vR = -abs(vL)
                        # Adjust bounds on vL and vR
                        vL = max(min(vL,VMAX),-VMAX)
                        vR = max(min(vR,VMAX),-VMAX)
                        if abs(vL)-1 < 0.15 and abs(vR)-1 < 0.15:
                            # If very close, halve the wheel speeds
                            vL /= 2
                            vR /= 2
                        print(vL, vR)
                else: # Detections is None
                    # Continue forward until cube is seen again
                    vL = 2
                    vR = 2
                    if timesteps_without_detection > 50:
                        # If cube hasnt been seen for a while, go back to following aisle path
                        TASK = "follow_aisle"
                        timesteps_without_detection = 0
                    else:
                        timesteps_without_detection += 1

            else: # Grab cube
                if cube_bounds is None:
                    cube_bounds = [117, 125, 126, 135] # Temp value for testing arm movement
                    print("Error: reaching for undefined cube!")
                # Go through arm state machine, moving the arm to the cube, grabbing it, and putting it in the basket
                if ARM_STATE == 0: # Start State
                    moveArmToTarget(armTopIk)
                    ARM_STATE = "go_to_shelf"
                    wait_timer = 0
                if ARM_STATE == "go_to_shelf":
                    if stalled_for(150):
                        x1, y1, x2, y2 = cube_bounds
                        x_offset = (x1 + x2) / 2 - 120
                        # Calculate which shelf the cube is on using camera height
                        shelf = UPPER_SHELF
                        if (y1 + y2) / 2 > 90:
                            shelf = LOWER_SHELF
                        shelfIk = calculateIk((0.4, 0, shelf))
                        moveArmToTarget(shelfIk)
                        openGrip()
                        ARM_STATE = "follow_arm_path"
                        # Calculate forward and side distances
                        FORWARD_DISTANCE = 0.68 if shelf == LOWER_SHELF else 0.75
                        SIDE_OFFSET = 0.01*x_offset-0.03
                        # Generate path to object
                        arm_linspace = np.linspace(np.array([0.4, SIDE_OFFSET]), np.array([FORWARD_DISTANCE, SIDE_OFFSET]), 5)
                        for point in arm_linspace:
                            arm_path.append((point[0], point[1], shelf))
                elif ARM_STATE == "follow_arm_path":
                    if stalled_for(75):
                        if arm_index >= len(arm_path):
                            ARM_STATE = "lower_arm"
                            arm_index = 0
                        else:
                            armForwardIk = calculateIk(arm_path[arm_index])
                            moveArmToTarget(armForwardIk)
                            arm_index += 1
                elif ARM_STATE == "lower_arm":
                    if stalled_for(150):
                        arm_x, arm_y, arm_z = arm_path[-1]
                        lowerIk = calculateIk((arm_x, arm_y, arm_z-0.15))
                        print("lowerIk")
                        moveArmToTarget(lowerIk)
                        ARM_STATE = "wiggle_cube"
                elif ARM_STATE == "wiggle_cube":
                    if stalled_for(150):
                        arm_x, arm_y, arm_z = arm_path[-1]
                        wiggleIk = calculateIk((arm_x+0.1, arm_y, arm_z-0.15))
                        print("wiggleIk")
                        moveArmToTarget(wiggleIk)
                        ARM_STATE = "close_grip"
                elif ARM_STATE == "close_grip":
                    if stalled_for(100):
                        closeGrip()
                        ARM_STATE = "wiggle_cube_2"
                elif ARM_STATE == "wiggle_cube_2":
                    if stalled_for(150):
                        arm_x, arm_y, arm_z = arm_path[-1]
                        wiggleIk = calculateIk((arm_x-0.05, arm_y, arm_z+0.05))
                        print("wiggleIk")
                        moveArmToTarget(wiggleIk)
                        ARM_STATE = "extract_from_shelf"
                elif ARM_STATE == "extract_from_shelf":
                    if stalled_for(100):
                        vL, vR = -3,-3
                        ARM_STATE = "move_to_basket"
                elif ARM_STATE == "move_to_basket":
                    if stalled_for(50):
                        vL, vR = 0,0
                        basketIk = calculateIk((0.2,0,0.8))
                        print("move_to_basket")
                        moveArmToTarget(basketIk)
                        ARM_STATE = "open_grip"
                elif ARM_STATE == "open_grip":
                    if stalled_for(100):
                        openGrip()
                        ARM_STATE = "done"
                elif ARM_STATE == "done":
                    ARM_STATE = 0
                    TASK = "follow_aisle"
                    WALL_MODE = "not_done"
                    cube_bounds = None

        # Actuator commands
        robot_parts[MOTOR_LEFT].setVelocity(vL)
        robot_parts[MOTOR_RIGHT].setVelocity(vR)
    

if MODE == 'map':
    file_path = os.path.join(os.getcwd(), "occupancy_grid.npy")
    np.save(file_path, occupancy_grid)
    print(f"[INFO] Occupancy grid saved to {file_path}")