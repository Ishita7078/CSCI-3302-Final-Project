"""grocery controller."""

# Apr 1, 2025

from controller import Robot
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import cv2
from scipy.signal import convolve2d
import random
from ultralytics import YOLO


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

keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

# MODE = 'map'
# MODE = 'planner'
MODE = 'navigation'


HEIGHT = 914
WIDTH = 494
SCALE = 30

map_size = (30, 16)  # meters
resolution = 0.0033  # meters per cell
grid_width = int(map_size[0] * SCALE)
grid_height = int(map_size[1] * SCALE)
occupancy_grid = np.zeros((grid_width, grid_height))
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)


def get_pose(gps, compass): #webots provided pose
    x_r = gps.getValues()[0]
    y_r = gps.getValues()[1]
    theta_r = np.arctan2(compass.getValues()[0], compass.getValues()[1])
    return x_r, -y_r, theta_r

def odometry():
    global vL, vR, pose_x, pose_y, pose_theta
    dt = timestep/1000

    vL = vL / MAX_SPEED * MAX_SPEED_MS
    vR = vR / MAX_SPEED * MAX_SPEED_MS

    dist_left = vL * dt
    dist_right = vR * dt
    # print(f'DIST_RIGHT {dist_right} DIST_LEFT {dist_left}')

    dist = (dist_right + dist_left) * 0.5
    theta = (dist_right - dist_left) / AXLE_LENGTH

    pose_x += dist * math.cos(pose_theta)
    pose_y += dist * math.sin(pose_theta)

    pose_theta += theta

    if pose_theta > math.pi:
        pose_theta -= 2 * math.pi
    elif pose_theta < -math.pi:
        pose_theta += 2 * math.pi

    return pose_x, pose_y, pose_theta


def to_pixels(x, y):
   return int((x + 15) * SCALE), int((y + 8.05) * SCALE)
    
  
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

def is_clear(lidar_vals, bin_range, threshold=2.0):
            return np.mean([lidar_vals[i] for i in bin_range]) > threshold

def grid_clear(pose_x, pose_y, pose_theta, direction="left", distance=0.5, size=3):
    angle_offset = math.pi / 2 if direction == "left" else -math.pi / 2
    check_theta = pose_theta + angle_offset

    check_x = pose_x + distance * math.cos(check_theta) #position in world coordinates
    check_y = pose_y + distance * math.sin(check_theta)
    
    x_pix, y_pix = to_pixels(check_x, check_y)

    half = size // 2 #to create grid around world
    for dx in range(-half, half + 1): #check grid around position
        for dy in range(-half, half + 1):
            x = x_pix + dx
            y = y_pix + dy
            if 0 <= x < grid_width and 0 <= y < grid_height:
                if occupancy_grid[x][y] == 3:  # obstacle
                    return False
    return True

def get_distance_helper(point1,point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def get_nearest_vertex(node_list, q_point):
    '''
    @param node_list: List of Node objects
    @param q_point: n-dimensional array representing a point
    @return Node in node_list with closest node.point to query q_point
    '''
    # TODO: Your Code Here
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
    '''
    @param from_point: n-Dimensional array (point) where the path to "to_point" is originating from (e.g., [1.,2.])
    @param to_point: n-Dimensional array (point) indicating destination (e.g., [0., 0.])
    @param delta_q: Max path-length to cover, possibly resulting in changes to "to_point" (e.g., 0.2)
    @returns path: list of points leading from "from_point" to "to_point" (inclusive of endpoints)  (e.g., [ [1.,2.], [1., 1.], [0., 0.] ])
    '''

    path = []

    # TODO: Figure out if you can use "to_point" as-is, or if you need to move it so that it's only delta_q distance away
    distance = get_distance_helper(from_point,to_point)
    if(distance > delta_q):
        direction = (to_point-from_point)/distance
        new_to_point = delta_q*direction+from_point
        line = np.linspace(from_point, new_to_point, num=10)
    else:
        line = np.linspace(from_point, to_point, num=10)

    # Convert to list of tuples of ints
    return [np.array((int(p[0]), int(p[1]))) for p in line]


def get_random_valid_vertex(state_valid, convolved_map):
    vertex = None
    while vertex is None: # Get starting vertex
        random_x = np.random.randint(0,WIDTH)
        random_y = np.random.randint(0,HEIGHT)
        pt = np.array((random_x,random_y))
        if state_valid(pt, convolved_map):
            vertex = pt
    return vertex


def near(node_list,q_new,r):
    near_list = []
    for node in node_list:
        distance = get_distance_helper(node.point,q_new)
        if(distance <=r): 
            near_list.append(node)
    return near_list


def state_is_valid(pt,convolved_map):
    x = int(pt[0])
    y = int(pt[1])
    if 0 <= x < WIDTH:
        if 0 <= y < HEIGHT:
            if(convolved_map[y,x] == 0):
                return True           
    return False


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


#------ RRT Star Helper --------#

if MODE == "planner":
    # start_w = (-7.97232, -4.84369) # (Pose_X, Pose_Y) in meters CHANGE
    # end_w = (-2.20815, -8.84167) # (Pose_X, Pose_Y) in meters

    # # Convert the start_w and end_w from the webots coordinate frame into the map frame

    # def world_to_map(coords):
    #     return (abs(int(coords[0]*30)), abs(int(coords[1]*30)))

    # start = world_to_map(start_w) # (x, y) in 360x360 map
    # end = world_to_map(end_w) # (x, y) in 360x360 map
    # print(start, end)


    class Node:
        def __init__(self, pt, parent=None):
                self.point = pt # n-Dimensional point
                self.parent = parent # Parent node
                self.path_from_parent = [] # List of points along the way from the parent node (for visualization)
            
    def rrt_star(convolved_map, state_is_valid, starting_point, goal_point, k, delta_q):
   
    #   RRT* Pseudo Code CREDIT: https://www.ri.cmu.edu/pub_files/2014/9/TR-2013-JDG003.pdf
        node_list = []
        cost_list = {}
        first = Node(starting_point, parent=None)
        node_list.append(first) # Add Node at starting point with no parent
        cost_list.update({tuple(first.point):0})
        rad = delta_q*1.5
        for i in range(1,k):
            if goal_point is not None and random.random() < 0.05: 
                q_rand = goal_point
            else:
                q_rand = get_random_valid_vertex(state_is_valid,convolved_map)
            q_nearest = get_nearest_vertex(node_list,q_rand)
            path_rand_nearest= steer(q_nearest.point,q_rand,delta_q)
            q_new = path_rand_nearest[-1]
            valid = True
            for point in path_rand_nearest:
                if state_is_valid(point,convolved_map) == False:
                    valid = False
            if valid:
                new_node = Node(q_new,q_nearest)
                new_node.path_from_parent = path_rand_nearest
                node_list.append(new_node)
                node_cost = cost_list[tuple(new_node.parent.point)]+ get_distance_helper(new_node.point, new_node.parent.point)
                cost_list.update({tuple(new_node.point):node_cost})
                q_near_list = near(node_list,new_node.point,rad)
                q_min = q_nearest
                c_min = cost_list.get(tuple(q_nearest.point)) + get_distance_helper(q_nearest.point,new_node.point)
                for node in q_near_list:
                    near_node_cost = cost_list[tuple(node.point)] + get_distance_helper(node.point,new_node.point)
                    if(near_node_cost<c_min):
                        node_path = np.linspace(node.point,new_node.point,num = 10)
                        path_valid = True
                        for point in node_path:
                            if state_is_valid(point,convolved_map) == False:
                                path_valid = False
                        if(path_valid):
                            q_min = node
                            c_min = near_node_cost
                new_node.parent = q_min
                new_node.path_from_parent = np.linspace(new_node.parent.point,new_node.point,num = 10)
                cost_list[tuple(new_node.point)] = c_min
                for node in q_near_list:
                    c_near = cost_list[tuple(node.point)]
                    c_new = cost_list[(tuple(new_node.point))] + get_distance_helper(node.point,new_node.point)
                    if(c_new < c_near): 
                        node_parent_path = np.linspace(new_node.point,node.point,num=10)
                        path_valid = True
                        for point in node_parent_path:
                            if state_is_valid(point,convolved_map) == False:
                                path_valid = False
                        if(path_valid):
                            node.parent = new_node
                            node.path_from_parent = np.linspace(new_node.point,node.point,num=10)
                            cost_list[tuple(node.point)] = c_new
                            

                if(goal_point is not None):
                    distance_from_goal = get_distance_helper(new_node.point,goal_point)
                    if(distance_from_goal < 1e-5):
                        print("Found the end!")
                        return node_list
        
        print(f"Did not find end, finished after {k} iterations.")
        return node_list
    
    #declare start and end and load map
    start = np.array((20,20))
    end = np.array((312,807))
    map = np.load("map.npy")
    map = map==3
   
    #Convolve map
    KERNEL_DIM = 15
    kernel = np.ones(shape=[KERNEL_DIM, KERNEL_DIM])
    convolved_map = convolve2d(map, kernel)
    convolved_map = convolved_map > 0
    convolved_map = convolved_map * 1
    height,width = convolved_map.shape

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
    
    print(path_waypoints)
    
    path_world_coords = []
    for point in path_waypoints:
        path_world_coords.append(((point[1]/30) - 15, (point[0] /30)-8.05))
    np.save("path.npy", path_world_coords)
    
    print(path_world_coords)

    path_map = np.load("path.npy")

    # x_coords, y_coords = zip(*waypoints_points)
    # plt.plot(x_coords, y_coords, color='red', linewidth=1)
    plt.plot(start[0], start[1], marker='^', color='lightgreen', markersize=7)
    plt.plot(end[0], end[1], marker='^', color='lightgreen', markersize=7)
    plt.show()
    visualize_2D_graph(convolved_map,waypoints_all,end,"test2.png")
# ------------------------------------------------------------------
# Helper Functions

gripper_status="closed"
state = "forward"

# Main Loop
while robot.step(timestep) != -1:
    pose_x, pose_y, pose_theta = odometry()
    lidar_values = np.array(lidar.getRangeImage())

    if MODE == 'map':
        # print(pose_x, pose_y, pose_theta)
        pose_x, pose_y, pose_theta = get_pose(gps, compass) #webots pose
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

        center_range = range(center_bin - 10, center_bin + 10)
        left_range   = range(left_bin - 10, left_bin + 10)
        right_range  = range(right_bin - 10, right_bin + 10)
        
        forward_clear = is_clear(lidar_values, center_range) 
        left_clear    = is_clear(lidar_values, left_range)
        right_clear   = is_clear(lidar_values, right_range) 

        if state == "forward":
            if forward_clear:
                vL, vR = 5, 5
            elif not grid_clear(pose_x, pose_y, pose_theta, "left"):
                state = "turn_right"
                turn_counter = 40
                vL, vR = 0, -3
            elif not grid_clear(pose_x, pose_y, pose_theta, "right"):
                state = "turn_left"
                turn_counter = 40
                vL, vR = -3, 0
            elif left_clear and right_clear:
                state = "turn_left"
                turn_counter = 40
                vL, vR = -3, 0
            else:
                state = "reverse"
                turn_counter = 20
                vL, vR = -4, -4
        
        elif state == "turn_left":
            if turn_counter > 0:
                vL, vR = -3, 0
                turn_counter -= 1
            else:
                state = "forward"
        
        elif state == "turn_right":
            if turn_counter > 0:
                vL, vR = 0, -3
                turn_counter -= 1
            else:
                state = "forward"
        
        elif state == "reverse":
            if turn_counter > 0:
                vL, vR = -4, -4
                turn_counter -= 1
            else:
                state = random.choice(["turn_left", "turn_right"])
                turn_counter = 20

        robot_parts["wheel_left_joint"].setVelocity(vL)
        robot_parts["wheel_right_joint"].setVelocity(vR)

        file_path = os.path.join(os.getcwd(), "occupancy_grid.npy")
        np.save(file_path, occupancy_grid)
        print(f"Occupancy grid saved to {file_path}")
    elif MODE == 'navigation':
        img = camera.getImageArray()
        img_np = np.array(img, dtype=np.uint8).reshape((height, width, 3))  #3 for RGB channels

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) #convert to BGR

        results = model(img_bgr)[0]  #inference
        detections = results.boxes.data.cpu().numpy()  #x1, y1, x2, y2, confidence, class
        class_names = model.names

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            label = class_names[int(cls)]
            print(f"[DETECTION] {label} ({conf:.2f}) at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")


    # if(gripper_status=="open"):
    #     # Close gripper, note that this takes multiple time steps...
    #     robot_parts["gripper_left_finger_joint"].setPosition(0)
    #     robot_parts["gripper_right_finger_joint"].setPosition(0)
    #     if right_gripper_enc.getValue()<=0.005:
    #         gripper_status="closed"
    # else:
    #     # Open gripper
    #     robot_parts["gripper_left_finger_joint"].setPosition(0.045)
    #     robot_parts["gripper_right_finger_joint"].setPosition(0.045)
    #     if left_gripper_enc.getValue()>=0.044:
    #         gripper_status="open"
    

if MODE == 'map':
    file_path = os.path.join(os.getcwd(), "occupancy_grid.npy")
    np.save(file_path, occupancy_grid)
    print(f"[INFO] Occupancy grid saved to {file_path}")