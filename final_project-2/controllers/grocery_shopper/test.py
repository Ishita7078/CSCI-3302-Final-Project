
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from scipy.signal import convolve2d # Uncomment if you want to use something else for finding the configuration space


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
# occupancy_grid = np.zeros((300, 300))
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
# lidar_offsets = lidar_offsets[50:len(lidar_offsets)-50] #provides clearest image

HEIGHT = 914
WIDTH = 494
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

if mode == "planner":
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
    map = np.load("final_project-2\controllers\grocery_shopper\map.npy")
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
