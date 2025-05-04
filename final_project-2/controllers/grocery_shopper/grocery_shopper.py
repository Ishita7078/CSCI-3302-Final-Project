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
robot = Supervisor()
marker = robot.getFromDef("marker").getField("translation") # TODO: remove
trans_field = robot.getSelf().getField("translation") # TODO: remove

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
    global wait_timer
    if wait_timer > timesteps:
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


def position_error(current_x, current_y, goal_x, goal_y):
    return math.sqrt((goal_x - current_x)**2 + (goal_y - current_y)**2)


def bearing_error(current_x, current_y, current_theta, goal_x, goal_y):
    y = goal_y - current_y
    x = goal_x - current_x
    alpha = math.atan2(y, x)
    alpha -= current_theta
    alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
    return alpha + math.pi/2


def heading_error(current_theta, goal_theta):
    eta = goal_theta - current_theta
    eta = (eta + math.pi) % (2 * math.pi) - math.pi
    return eta + math.pi/2


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
    if rho < 0.2:
        x_i = waypoints[index][0]
        y_i = waypoints[index][1]
        index += 1
    else:
        vL = (velocity - (angular * AXLE_LENGTH * 0.5)) / (0.02)
        vR = (velocity + (angular * AXLE_LENGTH * 0.5)) / (0.02)
        lsign = 1 if vL > 0 else -1
        rsign = 1 if vR > 0 else -1
        ratio = abs(vL / vR)
        if abs(vL) > MAX_SPEED or abs(vR) > MAX_SPEED:
            if ratio > 1:
                vL = MAX_SPEED * lsign
                vR = MAX_SPEED / ratio * rsign
            else:
                vL = MAX_SPEED * ratio * lsign
                vR = MAX_SPEED * rsign
        if abs(abs(vL / vR) - ratio) > 1e-10: exit()
        if alpha < -0.09:
            vL = (2)
            vR = (-2)
        elif alpha > 0.09:
            vL = (-2)
            vR = (2)

    if vL != 2 and vL != -2: vL = (vL / MAX_SPEED) * 2
    if vR != 2 and vR != -2: vR = (vR / MAX_SPEED) * 2

    # print(f"vL: {vL}, vR: {vR}")
    return vL, vR, x_i, y_i, index


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


def to_pixels_bad(x, y):
   return int((x + 15) * SCALE), int((y + 8.05) * SCALE)
   # return int((x + map_size[0]/2) * SCALE), int((y + map_size[1]/2 * SCALE))


def to_pixels(x,y):
    x = int((x + map_size[0]/2) / map_size[0] * HEIGHT)
    y = int((y + map_size[1]/2) / map_size[1] * WIDTH + 10)
    return x,y

def from_pixels_bad(x, y):
    return x / SCALE -15, y / SCALE - 8.05
    # return x / SCALE - map_size[0]/2, y / SCALE - map_size[1]/2

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

class Node:
    def __init__(self, pt, parent=None):
        self.point = pt  # n-Dimensional point
        self.parent = parent  # Parent node
        self.path_from_parent = []  # List of points along the way from the parent node (for visualization)


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

def rrt_star(convolved_map, state_is_valid, starting_point, goal_point, k, delta_q):
    #   RRT* Pseudo Code CREDIT: https://www.ri.cmu.edu/pub_files/2014/9/TR-2013-JDG003.pdf
    node_list = []
    cost_list = {}
    first = Node(starting_point, parent=None)
    node_list.append(first)  # Add Node at starting point with no parent
    cost_list.update({tuple(first.point): 0})
    rad = delta_q * 1.5
    for i in range(1, k):
        if goal_point is not None and random.random() < 0.05:
            q_rand = goal_point
        else:
            q_rand = get_random_valid_vertex(state_is_valid, convolved_map)
        q_nearest = get_nearest_vertex(node_list, q_rand)
        path_rand_nearest = steer(q_nearest.point, q_rand, delta_q)
        q_new = path_rand_nearest[-1]
        valid = True
        for point in path_rand_nearest:
            if state_is_valid(point, convolved_map) == False:
                valid = False
        if valid:
            new_node = Node(q_new, q_nearest)
            new_node.path_from_parent = path_rand_nearest
            node_list.append(new_node)
            node_cost = cost_list[tuple(new_node.parent.point)] + get_distance_helper(new_node.point,
                                                                                      new_node.parent.point)
            cost_list.update({tuple(new_node.point): node_cost})
            q_near_list = near(node_list, new_node.point, rad)
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
            new_node.parent = q_min
            new_node.path_from_parent = np.linspace(new_node.parent.point, new_node.point, num=10)
            cost_list[tuple(new_node.point)] = c_min
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

            if (goal_point is not None):
                distance_from_goal = get_distance_helper(new_node.point, goal_point)
                if (distance_from_goal < 1e-5):
                    print("Found the end!")
                    return node_list

    print(f"Did not find end, finished after {k} iterations.")
    print(f"Pixels: {(starting_point)}:{convolved_map[starting_point[1]][starting_point[0]]}, {(goal_point)}:{convolved_map[goal_point[1]][goal_point[0]]}")
    print(f"World: {from_pixels(starting_point)}, {from_pixels(goal_point)}")
    return node_list


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
def rotate_y(x, y, z, theta):
    new_x = x * np.cos(theta) + y * np.sin(theta)
    new_z = z
    new_y = y * -np.sin(theta) + x * np.cos(theta)
    return [-new_x, new_y, new_z]

def checkArmAtPosition(ikResults, cutoff=0.00005):
    # '''Checks if arm at position, given ikResults'''

    # Get the initial position of the motors
    initial_position = [0, 0, 0, 0] + [m.getPositionSensor().getValue() for m in motors] + [0, 0, 0, 0]

    # Calculate the arm
    arm_error = 0
    for item in builtins.range(14):
        arm_error += (initial_position[item] - ikResults[item]) ** 2
    arm_error = math.sqrt(arm_error)

    if arm_error < cutoff:
        if vrb:
            print("Arm at position.")
        return True
    return False


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
    initial_guess = []
    for i, link in enumerate(my_chain.links):
        if my_chain.active_links_mask[i]:
            lower, upper = link.bounds
            if lower != -np.inf and upper != np.inf:
                initial_guess.append((lower + upper) / 2.0)
            else:
                initial_guess.append(0.0)
        else:
            print(f"Setting {link} to 0")
            initial_guess.append(0.0)
    initial_guess = np.array(initial_guess)
    ik_result = my_chain.inverse_kinematics(
        target_position=np.array(target_position),
        target_orientation=target_orientation,
        orientation_mode=orientation_mode,
        initial_position=initial_guess
    )
    return ik_result



def getTargetFromObject(recognized_objects):
    # ''' Gets a target vector from a list of recognized objects '''

    # Get the first valid target
    target = recognized_objects[0].get_position()

    # Convert camera coordinates to IK/Robot coordinates
    # offset_target = [-(target[2])+0.22, -target[0]+0.08, (target[1])+0.97+0.2]
    offset_target = [-(target[2]) + 0.22, -target[0] + 0.06, (target[1]) + 0.97 + 0.2]

    return offset_target


def reachArm(target, previous_target, ikResults, cutoff=0.00005):
    # '''
    # This code is used to reach the arm over an object and pick it up.
    # '''

    # Calculate the error using the ikTarget
    error = 0
    ikTargetCopy = previous_target

    # Make sure ikTarget is defined
    if previous_target is None:
        error = 100
    else:
        for item in builtins.range(3):
            error += (target[item] - previous_target[item]) ** 2
        error = math.sqrt(error)

    # If error greater than margin
    if error > 0.05:
        print("Recalculating IK, error too high {}...".format(error))
        ikResults = calculateIk(target)
        ikTargetCopy = target
        moveArmToTarget(ikResults)

    # Exit Condition
    if checkArmAtPosition(ikResults, cutoff=cutoff):
        if vrb:
            print("NOW SWIPING")
        return [True, ikTargetCopy, ikResults]
    else:
        if vrb:
            print("ARM NOT AT POSITION")

    # Return ikResults
    return [False, ikTargetCopy, ikResults]


def closeGrip():
    robot.getDevice("gripper_right_finger_joint").setPosition(0.0)
    robot.getDevice("gripper_left_finger_joint").setPosition(0.0)

    # r_error = abs(robot.getDevice("gripper_right_finger_joint").getPositionSensor().getValue() - 0.01)
    # l_error = abs(robot.getDevice("gripper_left_finger_joint").getPositionSensor().getValue() - 0.01)

    # print("ERRORS")
    # print(r_error)
    # print(l_error)

    # if r_error+l_error > 0.0001:
    #     return False
    # else:
    #     return True


def openGrip():
    robot.getDevice("gripper_right_finger_joint").setPosition(0.045)
    robot.getDevice("gripper_left_finger_joint").setPosition(0.045)

    # r_error = abs(robot.getDevice("gripper_right_finger_joint").getPositionSensor().getValue() - 0.045)
    # l_error = abs(robot.getDevice("gripper_left_finger_joint").getPositionSensor().getValue() - 0.045)

    # if r_error+l_error > 0.0001:
    #     return False
    # else:
    #     return True



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

    #declare start and end and load map
    start = np.array((20,20))
    end = np.array((312,807))
    map = np.load("map.npy")
    map = map==3

    #Convolve map
    KERNEL_DIM = 50
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
# Helper Functions

aisle_path = [(-4.83, 5.82),(3.41, 5.82),(13.15, 5.82),(13.15,2.18),(3.41,2.18),(-4.83,2.18),(-4.83,-1.82),(3.41,-1.82),(13.15,-1.82),(13.15,-5.91),(3.41,-5.91),(-4.83,-5.91)]
# aisle_path_px = []
# for pos in aisle_path:
#     aisle_path_px.append(to_pixels(pos[0],pos[1]))

# Tiago Starting position
y_i = 0
x_i = -5

state = 0 # use this to iterate through your path
aisle_state = -1
current_path = []
ARM_STATE = 0

arm_path = []
arm_index = 0

TASK = "follow_aisle"
detection_timer = 0
timesteps_without_detection = 0
WALL_MODE = "not_done"
cube_bounds = None

UPPER_SHELF = 1.12
LOWER_SHELF = 0.7

cmap = np.load("convolved_map.npy")
# print(f"shape: {cmap.shape}")
# plt.imshow(cmap)
# plt.show()

armTop = (0,0,2)
armTopIk = calculateIk(armTop)

# Main Loop
while robot.step(timestep) != -1 and MODE != 'planner':
    print(f"WALL MODE: {WALL_MODE}, Wait Timer: {wait_timer}")
    # pose_x, pose_y, pose_theta = odometry()
    # print(pose_x, pose_y, pose_theta)
    lidar_values = np.array(lidar.getRangeImage())

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
    elif MODE == 'navigation':
        if current_path and len(current_path) > state:
            marker.setSFVec3f([current_path[state][0], current_path[state][1], 2.4]) # TODO: remove

        values = trans_field.getSFVec3f()
        pose_x = values[0]
        pose_y = values[1]
        n = compass.getValues()
        pose_theta = math.atan2(n[0], n[1])
        # TODO: remove ^^^^^^^^^
        # print(pose_x, pose_y, pose_theta)
        if WALL_MODE == "forward_left" or WALL_MODE == "forward_right":
            # Continue forward for x timesteps
            if WALL_MODE == "forward_left":
                vL, vR = 1.8, 2
            else:
                vL, vR = 2, 1.8
            print("Calling stalled_for")
            if stalled_for(150):
                if WALL_MODE == "forward_left":
                    WALL_MODE = "turn_left"
                elif WALL_MODE == "forward_right":
                    WALL_MODE = "turn_right"

            robot_parts[MOTOR_LEFT].setVelocity(vL)
            robot_parts[MOTOR_RIGHT].setVelocity(vR)
            continue
        elif WALL_MODE == "turn_left":
            vL = -2
            vR = 2
            if stalled_for(100):
                WALL_MODE = "done"
            robot_parts[MOTOR_LEFT].setVelocity(vL)
            robot_parts[MOTOR_RIGHT].setVelocity(vR)
            continue
        elif WALL_MODE == "turn_right":
            vL = 2
            vR = -2
            if stalled_for(100):
                WALL_MODE = "done"
            robot_parts[MOTOR_LEFT].setVelocity(vL)
            robot_parts[MOTOR_RIGHT].setVelocity(vR)
            continue

        detections = []
        if TASK != "grab_cube" and WALL_MODE not in ["forward_left", "forward_right", "turn_left", "turn_right"]:
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
        if len(detections) > 1: detections = [detections[0]]
        filtered_detections = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            # size = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if conf > 0.8 and (((x2-x1) > 4 and (y2-y1) > 4) or TASK == "go_to_cube"): # Confident it is a nearby cube
                label = class_names[int(cls)]
                if label == "yellow":
                    print(f"[DETECTION] {label} ({conf:.2f}) at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}], W {x2 - x1}, H {y2 - y1}")
                    filtered_detections.append(det)
        detections = filtered_detections

        if detections == [] and TASK == "follow_aisle":
            # task = go to next point in aisle path, no rrt star since just straight lines with no obstacles?
            # may run face-first into walls after picking up an object
            # if prev_state != state:
            # prev_state = state
            if state != len(current_path):
                # print(state, current_path)
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
                waypoints_all = rrt_star(cmap, state_is_valid, start, end, 1000, 10)
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
                plt.show()
        else: # detections is not None or TASK == "grab_cube"
            if TASK != "grab_cube":
                TASK = "go_to_cube"
                timesteps_without_detection = 0
                center_bin = LIDAR_ANGLE_BINS // 2
                print(f"Front lidar: {lidar_values[center_bin]}")
                largest_det = None
                largest_det_size = -1
                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    size = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if size > largest_det_size:
                        largest_det_size = size
                        largest_det = det
                if largest_det is not None:
                    x1, y1, x2, y2, _, _ = largest_det
                    x_offset = (x1+x2)/2 - 120
                    turn_speed = x_offset/40
                    turn_speed = max(min(turn_speed,1),-1)

                    min_lidar = 10
                    min_lidar_index = 0
                    for i in builtins.range(len(lidar_values)):
                        if lidar_values[i] > 0.6 and lidar_values[i] < min_lidar and i < len(lidar_values):
                            min_lidar = lidar_values[i]
                            min_lidar_index = i
                    print(f"Min Lidar: {min_lidar} at {min_lidar_index}")
                    if (min_lidar_index > 380 or min_lidar_index < 280) and (WALL_MODE == "not_done"):
                        # Wall following
                        if min_lidar < 1.5:
                            if min_lidar_index > 380 and x_offset > -10:
                                # Wall on right
                                print("following right")
                                print(pose_theta)
                                if 0 < pose_theta < 3:
                                    vL = 0
                                    vR = 2.5
                                else:
                                    print("forrrrrrrrrrewarod")
                                    vL = 1
                                    vR = 1
                                    if x_offset > 100:
                                        # TODO: continue for # timesteps, then turn with compass to face, then go to normal controller
                                        if math.pi/2 < pose_theta < math.pi or pose_theta < 0:
                                            print("turn hard right")
                                            vL = -1
                                            vR = 1
                                        else:
                                            print("done following wall")
                                            WALL_MODE = "forward_right"
                                robot_parts[MOTOR_LEFT].setVelocity(vL)
                                robot_parts[MOTOR_RIGHT].setVelocity(vR)
                                continue
                            elif min_lidar_index < 280 and x_offset < 10:
                                # Wall on left
                                print("following left")
                                if pose_theta < 0:
                                    vL = 2.5
                                    vR = 0
                                else:
                                    vL = 1
                                    vR = 1
                                    if x_offset < -100:
                                        print("Done following left")
                                        WALL_MODE = "forward_left"
                                robot_parts[MOTOR_LEFT].setVelocity(vL)
                                robot_parts[MOTOR_RIGHT].setVelocity(vR)
                                continue

                    if lidar_values[center_bin] < 1.11:
                        print("backing up")
                        vL = -1
                        vR = -1
                    elif 1.11 <= lidar_values[center_bin] <= 1.13 and abs(x_offset) < 3:
                        # Stop at wall
                        vL = 0
                        vR = 0
                        # Reach for cube
                        print("reaching")
                        TASK = "grab_cube"
                        cube_bounds = [x1, y1, x2, y2]
                    else:
                        VMAX = 2
                        speed = max(1, min(VMAX, (lidar_values[center_bin] - 1.13) * 5 + 1))
                        vL = speed * (1+turn_speed)
                        vR = speed * (1-turn_speed)
                        # Tight turn
                        if abs(vL) < 1 and abs(vR) > VMAX/2:
                            vL = -abs(vR)
                        elif abs(vR) < 1 and abs(vL) > VMAX/2:
                            vR = -abs(vL)
                        vL = max(min(vL,VMAX),-VMAX)
                        vR = max(min(vR,VMAX),-VMAX)
                        print(vL, vR)
                else: # Detections is None
                    # Continue forward until cube is seen again
                    vL = 2
                    vR = 2
                    print(f"DDDDDDDDDDDDDDDDDDDDDDDDDetections {detections}, {TASK}, {vL}, {vR}")
                    if timesteps_without_detection > 20:
                        #TODO: re-run RRT
                        TASK = "follow_aisle"
                    timesteps_without_detection += 1

            else: # Grab cube
                if cube_bounds is None:
                    cube_bounds = [117, 125, 126, 135] # Temp value for testing arm movement
                    print("Error: reaching for undefined cube!")
                if ARM_STATE == 0:
                    moveArmToTarget(armTopIk)
                    ARM_STATE = "go_to_shelf"
                    wait_timer = 0
                if ARM_STATE == "go_to_shelf":
                    if stalled_for(150):
                        x1, y1, x2, y2 = cube_bounds
                        shelf = UPPER_SHELF
                        if (y1 + y2) / 2 > 90:
                            shelf = LOWER_SHELF
                        shelfIk = calculateIk((0.4, 0, shelf))
                        moveArmToTarget(shelfIk)
                        openGrip()
                        ARM_STATE = "follow_arm_path"
                        # Reach forward distance based on object size and compass angle? size is bigger if looking at corner than face and upper vs lower shelf
                        FORWARD_DISTANCE = 0.68
                        SIDE_OFFSET = -0.03
                        arm_linspace = np.linspace(np.array([0.4, SIDE_OFFSET]), np.array([FORWARD_DISTANCE, SIDE_OFFSET]), 5)
                        for point in arm_linspace:
                            arm_path.append((point[0], point[1], shelf))
                elif ARM_STATE == "follow_arm_path":
                    if stalled_for(75):
                        if arm_index >= len(arm_path):
                            ARM_STATE = "lower_arm"
                            arm_index = 0
                        else:
                            print(arm_path[arm_index])
                            armForwardIk = calculateIk(arm_path[arm_index])
                            moveArmToTarget(armForwardIk)
                            arm_index += 1
                elif ARM_STATE == "lower_arm":
                    if stalled_for(150):
                        arm_x, arm_y, arm_z = arm_path[-1]
                        lowerIk = calculateIk((arm_x, arm_y, arm_z-0.15))
                        moveArmToTarget(lowerIk)
                        ARM_STATE = "wiggle_cube"
                elif ARM_STATE == "wiggle_cube":
                    if stalled_for(150):
                        arm_x, arm_y, arm_z = arm_path[-1]
                        wiggleIk = calculateIk((arm_x+0.1, arm_y, arm_z-0.15))
                        moveArmToTarget(wiggleIk)
                        ARM_STATE = "close_grip"
                elif ARM_STATE == "close_grip":
                    if stalled_for(100):
                        closeGrip()
                        ARM_STATE = "wiggle_cube_2"
                elif ARM_STATE == "wiggle_cube_2":
                    if stalled_for(150):
                        arm_x, arm_y, arm_z = arm_path[-1]
                        wiggleIk = calculateIk((arm_x-0.1, arm_y, arm_z-0.15))
                        moveArmToTarget(wiggleIk)
                        ARM_STATE = "raise_arm"
                elif ARM_STATE == "raise_arm":
                    if stalled_for(100):
                        raiseIk = calculateIk(arm_path[-1])
                        moveArmToTarget(raiseIk)
                        ARM_STATE = "extract_from_shelf"
                elif ARM_STATE == "extract_from_shelf":
                    if stalled_for(100):
                        extractIk = calculateIk((0.4, -0.4, shelf))
                        moveArmToTarget(extractIk)
                        ARM_STATE = "move_to_basket"
                elif ARM_STATE == "move_to_basket":
                    if stalled_for(150):
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
                #     if wait_timer > 200:
                #         armCubeIk = calculateIk((0.7,0,UPPER_SHELF+0.1),target_orientation=np.array([-1, 0.7, 0]),orientation_mode="Z")
                #         moveArmToTarget(armCubeIk)
                #         ARM_STATE = 3
                #     else:
                #         wait_timer += 1


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

        # Actuator commands
        robot_parts[MOTOR_LEFT].setVelocity(vL)
        robot_parts[MOTOR_RIGHT].setVelocity(vR)
    

if MODE == 'map':
    file_path = os.path.join(os.getcwd(), "occupancy_grid.npy")
    np.save(file_path, occupancy_grid)
    print(f"[INFO] Occupancy grid saved to {file_path}")