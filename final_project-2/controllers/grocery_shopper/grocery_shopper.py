"""""grocery controller."""

# Apr 1, 2025

from controller import Robot
import math
import numpy as np
from iterative_closest_point import icp_matching
import matplotlib.pyplot as plt
import os
import random



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

SCALE = 30

map_size = (30, 16)  # meters
resolution = 0.0033  # meters per cell
grid_width = int(map_size[0] * SCALE)
grid_height = int(map_size[1] * SCALE)
occupancy_grid = np.zeros((grid_width, grid_height))
state_bounds = np.array([[0,grid_width],[0,grid_height]])
# occupancy_grid = np.zeros((300, 300))
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
# lidar_offsets = lidar_offsets[50:len(lidar_offsets)-50] #provides clearest image

def get_pose(gps, compass): #webots provided pose, CHANGE
    x_r = gps.getValues()[0]
    y_r = gps.getValues()[1]
    theta_r = np.arctan2(compass.getValues()[0], compass.getValues()[1])
    return x_r, -y_r, theta_r

def rotate(ranges, angles): #rotate angles appropriately
    xs = ranges * np.cos(angles)
    ys = ranges * np.sin(angles)
    return np.vstack((xs, ys))


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

TURN_THRESHOLD = 0.05  # Allowable threshold for turning (in radians)

# Previous pose (to detect change in orientation)
last_pose_theta = 0

# Function to check if the robot is turning
def is_turning(pose_theta, last_pose_theta):
    if last_pose_theta is None:
        return False  # First step, no turning yet
    delta_theta = abs(pose_theta - last_pose_theta)
    return delta_theta > TURN_THRESHOLD  # Significant turn if delta exceeds threshold

# ------------------------------------------------------------------
# Helper Functions

poses = []  #list of (x, y, theta)
scans = []  #list of LIDAR
last_scan = None
last_pose = None

gripper_status="closed"
STEP = 100
step = 0
mode = "planner"


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
        return np.linspace(from_point,new_to_point,num=10)
    else:
        return np.linspace(from_point,to_point,num=10)
    
def get_random_valid_vertex(state_valid, bounds, obstacles):
    vertex = None
    while vertex is None: # Get starting vertex
        pt = np.random.rand(bounds.shape[0]) * (bounds[:,1]-bounds[:,0]) + bounds[:,0]
        if state_valid(pt):
            vertex = pt
    return vertex

def near(node_list,q_new,r):
    near_list = []
    for node in node_list:
        distance = get_distance_helper(node.point,q_new)
        if(distance <=r): 
            near_list.append(node)
    return near_list

    
#------ RRT Star Helper --------#

if mode == "planner":
    start_w = (-7.97232, -4.84369) # (Pose_X, Pose_Y) in meters CHANGE
    end_w = (-2.20815, -8.84167) # (Pose_X, Pose_Y) in meters

    # Convert the start_w and end_w from the webots coordinate frame into the map frame

    def world_to_map(coords):
        return (abs(int(coords[0]*30)), abs(int(coords[1]*30)))

    start = world_to_map(start_w) # (x, y) in 360x360 map
    end = world_to_map(end_w) # (x, y) in 360x360 map
    print(start, end)

    class Node:
        def __init__(self, pt, parent=None):
                self.point = pt # n-Dimensional point
                self.parent = parent # Parent node
                self.path_from_parent = [] # List of points along the way from the parent node (for visualization)
            
    def rrt_star(state_bounds, obstacles, state_is_valid, starting_point, goal_point, k, delta_q):
   
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
                q_rand = get_random_valid_vertex(state_is_valid,state_bounds,obstacles)
            q_nearest = get_nearest_vertex(node_list,q_rand)
            path_rand_nearest= steer(q_nearest.point,q_rand,delta_q)
            q_new = path_rand_nearest[-1]
            valid = True
            for point in path_rand_nearest:
                if not state_is_valid(point):
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
                            if not state_is_valid(point):
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
                            if not state_is_valid(point):
                                path_valid = False
                        if(path_valid):
                            node.parent = new_node
                            node.path_from_parent = np.linspace(new_node.point,node.point,num=10)
                            cost_list[tuple(node.point)] = c_new
                            

                if(goal_point is not None):
                    distance_from_goal = get_distance_helper(new_node.point,goal_point)
                    if(distance_from_goal < 1e-5):
                        return node_list

        return node_list
    map = np.load("map.npy")
    map = np.rot90(map, 3)
    
    plt.imshow(map, cmap='gray')
    plt.title("Map")
    plt.axis('off')
    plt.savefig("map_visualization.png")
   
    # Part 2.2: Compute an approximation of the “configuration space”
    KERNEL_DIM = 16
    kernel = np.ones(shape=[KERNEL_DIM, KERNEL_DIM])
    convolved_map = convolve2d(map, kernel)
    convolved_map = convolved_map > 0
    convolved_map = convolved_map * 1
    # plt.imshow(convolved_map)
    # plt.show()

    # Part 2.3 continuation: Call path_planner
    waypoints = rrt_star(state_bounds, state_is_valid, start, end, deltaq)

    # Part 2.4: Turn paths into waypoints and save on disk as path.npy and visualize it
    path_points = []  # normal path
    for point in waypoints:
        path_points.append((12 * (point[1] / -360), (12 * point[0] / -360)+0.3))
    np.save("path.npy", path_points)

    # path_map = np.load("path.npy")
    # plt.imshow(convolved_map)
    # y_coords, x_coords = zip(*waypoints)
    # plt.plot(x_coords, y_coords, color='red', linewidth=1)
    # plt.show()




# Main Loop
while robot.step(timestep) != -1:
    # print("Lidar max range:", lidar.getMaxRange())

        
    pose_x, pose_y, pose_theta = get_pose(gps, compass) #webots pose, CHANGE

    # print(pose_x, pose_y, pose_theta)
    lidar_values = np.array(lidar.getRangeImage())

    # turning = is_turning(pose_theta, last_pose_theta)
    # # turning = False
    # if step == STEP:
    #     last_pose_theta = pose_theta
    #     step = 0
    # else:
    #     print(step)
    #     step += 1

    # if not turning:

    pose_pixels = to_pixels(pose_x, pose_y)

    occupancy_grid[pose_pixels[0]][pose_pixels[1]] = 2

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
                if distance < 5 and distance > 0:
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
    # plt.clf()
    # plt.imshow(color_grid, origin="lower")
    # plt.draw()  # Update the plot
    # plt.pause(0.01)  # Keep the plot updated  


    # plt.clf()
    # plt.imshow(occupancy_grid, cmap="gray", origin="lower")
    # plt.title("Occupancy Grid Map (30x16m)")
    # plt.pause(0.01)
    center_bin = LIDAR_ANGLE_BINS // 2
    left_bin = int(center_bin - 100)
    right_bin = int(center_bin + 100)

    # # Thresholds for "clear"
    forward_clear = lidar_values[center_bin] > 1.0
    left_clear = lidar_values[left_bin] > 1.0
    right_clear = lidar_values[right_bin] > 1.0

    # Finite State Machine for obstacle avoidance
    if forward_clear:
        vL, vR = 10, 10  # move forward
    # elif left_clear and not right_clear:
    #     vL, vR = 2.5, -2.5  # turn left
    # elif right_clear and not left_clear:
    #     vL, vR = -2.5, 2.5  # turn right
    # elif left_clear and right_clear:
    #     vL, vR = 2.5, -2.5  # both sides open? turn left arbitrarily
    else:
        vL, vR = -1, 1  # if boxed in, turn right


    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)

    file_path = os.path.join(os.getcwd(), "occupancy_grid.npy")
    np.save(file_path, occupancy_grid)
    print(f"[INFO] Occupancy grid saved to {file_path}")
    
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

