"""
## Additional supplementary experiments
We also make an additional attempt to integrate the idea of multimodal environment information with the A* algorithm, 
where we learn the historical optimal data through a multi-layer perceptron network, replacing the original heuristic function h.
There is a slight improvement in the results.
And Data-driven can combine multimodal environment information with more path planning algorithms (such as A*).
"""
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
show_animation = True


#The following data is the meteorological information corresponding to the test area and can be replaced
excel_file_path1 = r'Meteorological analysis module\Meterorological Data\u2_new.xlsx'
excel_file_path2 = r'Meteorological analysis module\Meterorological Data\v2_new.xlsx'
df1 = pd.read_excel(excel_file_path1)
df2 = pd.read_excel(excel_file_path2)
array1 = df1.iloc[4:16, 12:24].values#21*37  5*5  12*12
array2 = df2.iloc[4:16, 12:24].values
list_u2 = [[round(num, 5) for num in row] for row in array1]
list_v2 = [[round(num, 5) for num in row] for row in array2]
u2 = np.loadtxt(r"Path planning module\Experiments for Comparison\A_star\meteorological data\u2.txt")
v2 = np.loadtxt(r"Path planning module\Experiments for Comparison\A_star\meteorological data\v2.txt")

#Vector field - ocean current field
#The following data is the meteorological information corresponding to the test area and can be replaced
excel_file_path_u = r'Meteorological analysis module\Meterorological Data\u_interpolated.xlsx'
excel_file_path_v = r'Meteorological analysis module\Meterorological Data\v_interpolated.xlsx'
df_u = pd.read_excel(excel_file_path_u)
df_v = pd.read_excel(excel_file_path_v)
array_u = df_u.iloc[4:16, 12:24].values
array_v = df_v.iloc[4:16, 12:24].values
list_u = [[round(num, 5) for num in row] for row in array_u]
list_v = [[round(num, 5) for num in row] for row in array_v]
u = np.loadtxt(r"Path planning module\Experiments for Comparison\A_star\meteorological data\u.txt")
v = np.loadtxt(r"Path planning module\Experiments for Comparison\A_star\meteorological data\v.txt")



class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(7, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
model = MLP()
model.load_state_dict(torch.load(r'Path planning module\Experiments for Comparison\A_star\mlp_model.pth'))  

class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr, angle_threshold=36):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m],
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.angle_threshold = angle_threshold
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search
        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node
        prev_angle =  math.degrees(math.atan((gy-sy)/(gx-sx)))
        angle = prev_angle

        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(open_set[o].x, open_set[o].y,
                                                                     goal_node.x, goal_node.y, 0, 
                                                                     u[goal_node.x, 66-goal_node.y],
                                                                     v[goal_node.x, 66-goal_node.y], 
                                                                     u2[goal_node.x, 66-goal_node.y],
                                                                     v2[goal_node.x, 66-goal_node.y]))
            current = open_set[c_id]

            # # show graph
            # if show_animation:  # pragma: no cover
            #     plt.plot(self.calc_grid_position(current.x, self.min_x),
            #              self.calc_grid_position(current.y, self.min_y), "xc")
            #     # for stopping simulation with the esc key.
            #     plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            #     if len(closed_set.keys()) % 10 == 0:
            #         plt.pause(0.001)

            if math.hypot(current.x - goal_node.x, current.y - goal_node.y) <= 10:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                radian = math.radians(self.motion[i][1]+angle)
                dx = math.cos(radian)*7 #The speed after conversion (which can be adjusted based on grid size.)
                dy = math.sin(radian)*7
                node = self.Node(round(current.x + dx + u2[current.x, 66-current.y] + u[current.x, 66-current.y]),
                                 round(current.y + dy + v2[current.x, 66-current.y] + v[current.x, 66-current.y]),
                                 current.cost + self.motion[i][0], c_id)

                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue
                # Check for turning angle constraint


                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node
                angle = (self.motion[i][0]+angle) % 360

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(gx,gy,x, y, angle, u, v, u2, v2):
        w = 0.9 # weight of multimodal environmental infomation
        d = (1-w)* math.hypot(gx - x, gy - y) + w * model(torch.tensor([x, y, angle, u, v, u2, v2], dtype=torch.float32)).item()
        #d =  model(torch.tensor([x, y, angle, u, v, u2, v2], dtype=torch.float32)).item()
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position
        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        # print("min_x:", self.min_x)
        # print("min_y:", self.min_y)
        # print("max_x:", self.max_x)
        # print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        # print("x_width:", self.x_width)
        # print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        # motion = [[1, 0, 1],
        #           [0, 1, 1],
        #           [-1, 0, 1],
        #           [0, -1, 1],
        #           [-1, -1, math.sqrt(2)],
        #           [-1, 1, math.sqrt(2)],
        #           [1, -1, math.sqrt(2)],
        #           [1, 1, math.sqrt(2)]]
        motion = []
        for a in [-36,-18,0,18,36]:
            cost = 1  
            angle_motion = a
            motion.append([cost, angle_motion])

        return motion
    


def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = 4  # [m]   #sx,xy the position of starting point
    sy = 7 # [m]
    gx = 60  # [m]  #gx,gy the position of the goal point
    gy = 73  # [m]
    grid_size = 1 # [m] we need to adjust the map resolution appropriately.
    robot_radius = 2.0  # [m]

    # # set obstacle positions
    # ox, oy = [], []
    # for i in range(-10, 60):
    #     ox.append(i)
    #     oy.append(-10.0)
    # for i in range(-10, 60):
    #     ox.append(60.0)
    #     oy.append(i)
    # for i in range(-10, 61):
    #     ox.append(i)
    #     oy.append(60.0)
    # for i in range(-10, 61):
    #     ox.append(-10.0)
    #     oy.append(i)
    # for i in range(-10, 40):
    #     ox.append(20.0)
    #     oy.append(i)
    # for i in range(0, 40):
    #     ox.append(40.0)
    #     oy.append(60.0 - i)
    output_file_path = r'Path planning module\Experiments for Comparison\A_star\grid_with_circles.txt'
    grid_with_circles = np.loadtxt(output_file_path, dtype=int)
    # print(grid_with_circles.shape[0])
    # print(grid_with_circles.shape[1])
    obstacle_x_coordinates = []
    obstacle_y_coordinates = []
    for y in range(grid_with_circles.shape[0]):
        for x in range(grid_with_circles.shape[1]):
            if grid_with_circles[y, x] == 0:
                obstacle_x_coordinates.append(x)
                obstacle_y_coordinates.append(80-y)
    ox = obstacle_x_coordinates
    oy = obstacle_y_coordinates

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")
        
        

    # start_time = time.time()  
    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)
    end_time = time.time()  
    # dtime = end_time - start_time
    # print("running time: %.8s s" % dtime)  


    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()


if __name__ == '__main__':
    main()

