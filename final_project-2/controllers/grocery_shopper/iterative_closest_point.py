# """
# Iterative Closest Point (ICP) SLAM example
# author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı, Shamil Gemuev
# """

import math

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np

#  ICP parameters
EPS = 0.0001
MAX_ITER = 100

show_animation = True

map_size = (30, 16)  # meters
resolution = 0.0166  # meters per cell
grid_width = int(map_size[0] / resolution)
grid_height = int(map_size[1] / resolution)
occupancy_grid = np.zeros((grid_height, grid_width))
print(occupancy_grid.shape[0], occupancy_grid.shape[1])


def icp_matching(previous_points, current_points):
    global map_size, resolution, grid_width, grid_height, occupancy_grid
    """
    Iterative Closest Point matching
    - input
    previous_points: 2D or 3D points in the previous frame
    current_points: 2D or 3D points in the current frame
    - output
    R: Rotation matrix
    T: Translation vector
    """
    H = None  # homogeneous transformation matrix

    dError = np.inf
    preError = np.inf
    count = 0

    if show_animation:
        fig = plt.figure()
        if previous_points.shape[0] == 3:
           fig.add_subplot(111, projection='3d')

    while dError >= EPS:
        count += 1

        if show_animation:  # pragma: no cover
            plot_points(previous_points, current_points, fig)
            plt.pause(0.1)

        indexes, error = nearest_neighbor_association(previous_points, current_points)
        Rt, Tt = svd_motion_estimation(previous_points[:, indexes], current_points)
        # update current points
        current_points = (Rt @ current_points) + Tt[:, np.newaxis]

        dError = preError - error
        # print("Residual:", error)

        if dError < 0:  # prevent matrix H changing, exit loop
            # print("Not Converge...", preError, dError, count)
            break

        preError = error
        H = update_homogeneous_matrix(H, Rt, Tt)

        if dError <= EPS:
            # print("Converge", error, dError, count)
            break
        elif MAX_ITER <= count:
            # print("Not Converge...", error, dError, count)
            break

    R = np.array(H[0:-1, 0:-1])
    T = np.array(H[0:-1, -1])

    return R, T


def update_homogeneous_matrix(Hin, R, T):


    r_size = R.shape[0]
    H = np.zeros((r_size + 1, r_size + 1))

    H[0:r_size, 0:r_size] = R
    H[0:r_size, r_size] = T
    H[r_size, r_size] = 1.0

    if Hin is None:
        return H
    else:
        return Hin @ H


def nearest_neighbor_association(previous_points, current_points):

    # calc the sum of residual errors
    delta_points = previous_points - current_points
    d = np.linalg.norm(delta_points, axis=0)
    error = sum(d)

    # calc index with nearest neighbor assosiation
    d = np.linalg.norm(np.repeat(current_points, previous_points.shape[1], axis=1)
                       - np.tile(previous_points, (1, current_points.shape[1])), axis=0)
    indexes = np.argmin(d.reshape(current_points.shape[1], previous_points.shape[1]), axis=1)

    return indexes, error


def svd_motion_estimation(previous_points, current_points):
    pm = np.mean(previous_points, axis=1)
    cm = np.mean(current_points, axis=1)

    p_shift = previous_points - pm[:, np.newaxis]
    c_shift = current_points - cm[:, np.newaxis]

    W = c_shift @ p_shift.T
    u, s, vh = np.linalg.svd(W)

    R = (u @ vh).T
    t = pm - (R @ cm)

    return R, t

def world_to_grid(x, y):
    # Origin in the middle of the grid
    print(f"x and y: {x} {y}")
    global map_size, resolution, grid_width, grid_height, occupancy_grid
    # Convert to grid coordinates
    grid_x = int((x+15) / resolution)
    grid_y = int((y+8.05) / resolution)  # Flip Y to match image coordinate system
    print(f"grid_x and grid_y: {grid_x} {grid_y}")

    return grid_x, grid_y

# def plot_points(previous_points, current_points, figure=None): 
#     global map_size, resolution, grid_width, grid_height, occupancy_grid
#     # Update the occupancy grid with current_points
#     for i in range(current_points.shape[1]):
#         x, y = current_points[0, i], current_points[1, i]
#         coords = world_to_map_coords(x, y)
#         if coords:
#             mx, my = coords
#             occupancy_grid[my, mx] = 1  # Mark as occupied

#     # Plot the full map once in a persistent window
#     plt.clf()
#     plt.imshow(occupancy_grid, cmap="gray", origin="lower")
#     plt.title("Occupancy Grid Map (30x16m)")
#     plt.pause(0.01)

# def plot_points(previous_points, current_points, figure=None):
#     global map_size, resolution, grid_width, grid_height, occupancy_grid

#     robot_map_pos = world_to_grid(0.0, 0.0)  # robot is at (0, 0)
#     if not robot_map_pos:
#         return
#     rx, ry = robot_map_pos

#     for i in range(current_points.shape[1]):
#         x, y = current_points[0, i], current_points[1, i]
#         coords = world_to_grid(x, y)
#         if coords is not None:
#             mx, my = coords
#             # Trace the line to the obstacle
#             line_algo(rx, ry, mx, my)
#             # Mark endpoint as occupied (obstacle)
#             occupancy_grid[my][mx] = 0.5

#     # Plot map
#     plt.clf()
#     plt.imshow(occupancy_grid, cmap="gray", origin="lower")
#     plt.title("Occupancy Grid Map (30x16m)")
#     plt.pause(0.01)

def line_algo(x0, y0, x1, y1):  # bresenham's line algorithm
    global occupancy_grid
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x_inc = 1 if x0 < x1 else -1
    y_inc = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if 0 <= y0 < occupancy_grid.shape[0] and 0 <= x0 < occupancy_grid.shape[1]:
            if occupancy_grid[y0][x0] != 1:
                occupancy_grid[y0][x0] = 0.2  # mark as free
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += x_inc
        if e2 < dx:
            err += dx
            y0 += y_inc



def plot_points(previous_points, current_points, figure):
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])
    if previous_points.shape[0] == 3:
        plt.clf()
        axes = figure.add_subplot(111, projection='3d')
        axes.scatter(previous_points[0, :], previous_points[1, :],
                    previous_points[2, :], c="r", marker=".")
        axes.scatter(current_points[0, :], current_points[1, :],
                    current_points[2, :], c="b", marker=".")
        axes.scatter(0.0, 0.0, 0.0, c="r", marker="x")
        figure.canvas.draw()
    else:
        plt.cla()
        plt.plot(previous_points[0, :], previous_points[1, :], ".r")
        plt.plot(current_points[0, :], current_points[1, :], ".b")
        plt.plot(0.0, 0.0, "xr")
        plt.axis("equal")


def main():
    # print(__file__ + " start!!")

    # simulation parameters
    nPoint = 1000
    fieldLength = 50.0
    motion = [0.5, 2.0, np.deg2rad(-10.0)]  # movement [x[m],y[m],yaw[deg]]

    nsim = 3  # number of simulation



    for _ in range(nsim):

        # previous points
        px = (np.random.rand(nPoint) - 0.5) * fieldLength
        py = (np.random.rand(nPoint) - 0.5) * fieldLength
        previous_points = np.vstack((px, py))

        # current points
        cx = [math.cos(motion[2]) * x - math.sin(motion[2]) * y + motion[0]
              for (x, y) in zip(px, py)]
        cy = [math.sin(motion[2]) * x + math.cos(motion[2]) * y + motion[1]
              for (x, y) in zip(px, py)]
        current_points = np.vstack((cx, cy))

        R, T = icp_matching(previous_points, current_points)
        # print("R:", R)
        # print("T:", T)


def main_3d_points():
    # print(__file__ + " start!!")

    # simulation parameters for 3d point set
    nPoint = 1000
    fieldLength = 50.0
    motion = [0.5, 2.0, -5, np.deg2rad(-10.0)]  # [x[m],y[m],z[m],roll[deg]]

    nsim = 3  # number of simulation

    for _ in range(nsim):

        # previous points
        px = (np.random.rand(nPoint) - 0.5) * fieldLength
        py = (np.random.rand(nPoint) - 0.5) * fieldLength
        pz = (np.random.rand(nPoint) - 0.5) * fieldLength
        previous_points = np.vstack((px, py, pz))

        # current points
        cx = [math.cos(motion[3]) * x - math.sin(motion[3]) * z + motion[0]
              for (x, z) in zip(px, pz)]
        cy = [y + motion[1] for y in py]
        cz = [math.sin(motion[3]) * x + math.cos(motion[3]) * z + motion[2]
              for (x, z) in zip(px, pz)]
        current_points = np.vstack((cx, cy, cz))

        R, T = icp_matching(previous_points, current_points)
        # print("R:", R)
        # print("T:", T)


if __name__ == '__main__':
    main()
    main_3d_points()

