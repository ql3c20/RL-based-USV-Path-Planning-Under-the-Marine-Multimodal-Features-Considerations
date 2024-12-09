"""
@File  : fusion_DQN.py
@Author: DavidLin
@Date  : 2024/12/7
@Contact : davidlin659562@gmail.com
@Description : 
This is the environment construction file for this article "RL-based USV Path Planning Under the Marine Multimodal Features Considerations"
state space = [ppx, ppy, angle, tx, ty, u2_wind, v2_wind, u_ocean, v_ocean]
action space = [-36°, -18°, 0°, 18°, 36°]
"""
import matplotlib.pyplot as plt
import math
import gym  
from gym import spaces 
from gym.utils import seeding 
import numpy as np
from gym.envs.classic_control import rendering
from scipy.spatial import ConvexHull
import netCDF4 as nc
import pandas as pd
from txt_to_matrix import read_binary_txt 
# from opencv import center_change, radii
import pandas as pd
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#The following random obstacle data can be replaced by Edge detection and Contour extraction results
# The position of obstacles is theoretically randomly generated
center_change = [(26, 175), (122, 102), (140, 161), (129, 56), (106, 10), (215, 154), (200, 16), (105, 33)]
radii = [2, 4, 2, 3, 2, 2, 4, 3]


RAD2DEG = 57.29577951308232  
CHANGE = False



#----------------------------Multimodal environment information fusion---------------------------------------------
#The images below are test images and can be replaced with maps of different areas
file_path = r'Image processing module\Geographic Data\testmap_105108.txt' 
binary_grid_reverse = read_binary_txt(file_path)
binary_array_reverse = np.array(binary_grid_reverse)
binary_grid = np.flipud(binary_array_reverse)


#Vector field - wind field
#The following data is the meteorological information corresponding to the test area and can be replaced
excel_file_path1 = r'Meteorological analysis module\Meterorological Data\u10_new.xlsx'
excel_file_path2 = r'Meteorological analysis module\Meterorological Data\v10_new.xlsx'
df1 = pd.read_excel(excel_file_path1)
df2 = pd.read_excel(excel_file_path2)
array1 = df1.iloc[4:16, 12:24].values#21*37  5*5  12*12
array2 = df2.iloc[4:16, 12:24].values
list_u10 = [[round(num, 5) for num in row] for row in array1]
list_v10 = [[round(num, 5) for num in row] for row in array2]



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





class PuckWorldEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    #The following hyperparameters need to be adjusted based on the selected scenario.
    def __init__(self):
        self.width = 300 #screen width
        self.length = 300  #screen length
        self.speed = 7
        self.accel_x = 1 # agent acceleration
        self.accel_y = 1
        self.interval = 400
        self.goal_dis = 10  # expected goal distance 
        self.t = 0  # puck world clock
        self.update_time = 1  # time for target randomize its position
        self.circles_this = center_change
        self.radius_this = radii
        self.low = np.array([0,  # agent position x
                             0,
                             -np.inf,
                             0, # target position x
                             0,
                             -np.inf,
                             -np.inf,
                             -np.inf,
                             -np.inf
                             ])
        self.high = np.array([300,
                              300,
                              np.inf,
                              300,
                              300,
                              np.inf,
                              np.inf,
                              np.inf,
                              np.inf
                              ])

        self.reward = 0  # for rendering
        self.collision = 0
        self.action = None  # for rendering
        self.viewer = None
        self.done = None
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(self.low, self.high)#state space
        self.reset()


    def step(self, action):
        

        action = int(action)#GPU
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        self.reward = 0   
        self.collision = 0 

  
        self.action = action  # action for rendering
        ppx, ppy, angle, tx, ty, u2_right, v2_right, u_right, v_right= self.state  

        angle_last = angle
        angle_last = round(angle_last,5)
        #action space--------------------------------
        if action == 0: 
            angle -= 36 
            angle = angle % 360  
        if action == 1: 
            angle -= 18
            angle = angle % 360
        if action == 2: 
            angle = angle
            angle = angle % 360
        if action == 3: 
            angle += 18
            angle = angle % 360 
        if action == 4:
            angle +=36
            angle = angle % 360


        #Original position
        ppx_last = ppx
        ppx_last = round(ppx_last,5)
        ppy_last = ppy 
        ppy_last = round(ppy_last,5)

        

    
        #update agent position
        pvx = self.speed * self.interval / 333000 * 300 * math.cos(math.radians(angle))#Converted to grid velocity
        pvx = round(pvx,5)
        pvy = self.speed * self.interval / 333000 * 300 * math.sin(math.radians(angle))
        pvy = round(pvy,5)
        ppx += pvx + u_right + u2_right
        ppx = round(ppx,5)
        ppy += pvy + v_right + v2_right # update agent position
        ppy = round(ppy,5)


        dx, dy = ppx - tx, ppy - ty  # calculate distance from
        dis = self.compute_dis(abs(dx), abs(dy))  # agent to target
        self.reward += round((self.goal_dis - dis)/100, 5)  # give an reward



        self.done = bool(dis <= self.goal_dis) 
        if self.done :
            self.reward += 100
            return self.state, self.reward, self.done, self.collision



        #Detect collisions with edges
        if ppx <= 90:  # encounter left bound. Adjust according to task requirements
            ppx = ppx_last
            ppy = ppy_last
            angle = 540 - angle_last
            angle = angle % 360
            self.reward -= 10
            #self.reward -= 0.1~10
            # Penalty value can be adjusted basd on Select scenarios and task requirements
            self.collision = 1
            return self.state, self.reward, self.done, self.collision

        if ppx >= 155:  # right bound. Adjust according to task requirements
            ppx = -ppx_last
            ppy = ppy_last
            angle = 540 - angle_last
            angle = angle % 360
            self.reward -= 10
            #self.reward -= 0.1~10
            # Penalty value can be adjusted basd on Select scenarios and task requirements
            self.collision = 1

            return self.state, self.reward, self.done, self.collision

        if ppy <= 17:  # bottom bound. Adjust according to task requirements
            ppx = ppx_last
            ppy = ppy_last
            angle = 360 - angle_last
            angle = angle % 360
            self.reward -= 10
            #self.reward -= 0.1~10
            # Penalty value can be adjusted basd on Select scenarios and task requirements
            self.collision = 1

            return self.state, self.reward, self.done, self.collision
        
        if ppy >= 97:  # top bound. Adjust according to task requirements
            ppx = ppx_last
            ppy = ppy_last
            angle = 360 -angle_last
            angle = angle % 360
            self.reward -= 10
            self.collision = 1
            #self.reward -= 0.1~10
            # Penalty value for detect collisions with edges can be adjusted basd on Select scenarios and task requirements
            return self.state, self.reward, self.done, self.collision

        def point_to_line_distance(point, line_coefficients):
            x0, y0 = point
            a, b, c = line_coefficients
            numerator = abs(a * x0 + b * y0 + c)
            denominator = math.sqrt(a**2 + b**2)
            distance = numerator / denominator
            return distance
        def line_coefficients(point1, point2):
            x1, y1 = point1
            x2, y2 = point2
            m = (y2 - y1) / (x2 - x1)
            b = -m * x1 + y1
            a = -m
            c = -b
            return a, 1, c
        def linear_interpolation(point1, point2, num_points):
            x1, y1 = point1
            x2, y2 = point2
            x_values = [x1 + (x2 - x1) * i / (num_points - 1) for i in range(num_points)]
            y_values = [y1 + (y2 - y1) * i / (num_points - 1) for i in range(num_points)]
            interpolated_points = list(zip(x_values, y_values))
            return interpolated_points


        #Detect collisions with shorelines
        a,b,c = line_coefficients((ppx,ppy), (ppx_last,ppy_last))
        num_points = 10
        interpolated_points = linear_interpolation((ppx,ppy), (ppx_last,ppy_last), num_points)
        for p in interpolated_points:
            if binary_grid[math.ceil(p[1])][math.ceil(p[0])] == 0:   #碰到障碍
                ppx = ppx_last
                ppy = ppy_last
                angle = angle_last
                self.collision = 1
                self.reward -= 10
                #self.reward -= 0.1~10
                #Penalty value for detect collisions with shorelines can be adjusted basd on Select scenarios and task requirements
                return self.state, self.reward, self.done, self.collision

                
        #Detect collisions with circle obstacles-----------------------------------------------------------------------
        for i in self.circles_this:
            distance_ = point_to_line_distance((i[0],i[1]),(a,b,c))
            radius_ = self.radius_this[self.circles_this.index(i)]
            if distance_<=radius_:
                ppx = ppx_last
                ppy = ppy_last
                angle = angle_last
                self.collision = 1
                self.reward -= 10
                #Penalty value for detect collisions with shorelines can be adjusted basd on Select scenarios and task requirements
                return self.state, self.reward, self.done, self.collision



 

        row_index_10 = int(3*(1-ppy/300)//0.25) 
        column_index_10 = int(ppx/300*3//0.25) 

        u10 = list_u10[row_index_10][column_index_10]#x
        v10 = list_v10[row_index_10][column_index_10]#y
        u2 = u10 * (0.4*math.log(2/0.003) / math.log(10/0.003))#speed
        v2 = v10 * (0.4*math.log(2/0.003) / math.log(10/0.003))
        u2_right = round(u2*self.interval/333000*300, 5) 
        v2_right = round(v2*self.interval/333000*300, 5)

        
        
        row_index = int(3*(1-ppy/300)//0.25)
        column_index = int(ppx/300*3//0.25) 
        u = list_u[row_index][column_index]#x
        v = list_v[row_index][column_index]#y
        u_right = round(u*self.interval/333000*300, 5)
        v_right = round(v*self.interval/333000*300, 5)

        
        self.state = (ppx, ppy, angle, tx, ty, u2_right, v2_right, u_right, v_right)#9-dimension state spce
        
        # print(self.collision)
        return self.state, self.reward, self.done, self.collision


    def compute_dis(self, dx, dy):
        return math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))


    def reset(self):
        ppx = 95
        ppy = 22
        tx = 149
        ty = 92
        angle = math.degrees(math.atan((ty - ppy) / (tx - ppx)))
        row_index_10 = int(3*(1-ppy/300)//0.25) 
        column_index_10 = int(ppx/300*3//0.25) 
        u10 = list_u10[row_index_10][column_index_10]#x direction
        v10 = list_v10[row_index_10][column_index_10]#y direction
        u2 = u10 * (math.log(2/0.003) / math.log(10/0.003))
        v2 = v10 * (math.log(2/0.003) / math.log(10/0.003))
        u2_right = round(u2*self.interval/333000*300, 5) 
        v2_right = round(v2*self.interval/333000*300, 5)
        row_index = int(3*(1-ppy/300)//0.25) 
        column_index = int(ppx/300*3//0.25) 
        u = list_u[row_index][column_index]#x direction
        v = list_v[row_index][column_index]#y direction
        u_right = round(u*self.interval/333000*300, 5)
        v_right = round(v*self.interval/333000*300, 5)
        self.state = np.array([ppx, ppy, angle, tx, ty, u2_right, v2_right, u_right, v_right])
        #self.state = [330, 330, angle, 805, 575].clone().detach().to(device, dtype=torch.float32)
        # self.state = torch.tensor([ppx, ppy, pvx, pvy, tx, ty], device=device, dtype=torch.float32)
        return self.state 

    def render(self,x,y, mode='human', close=False,DRAW=False):
        if close: 
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        
        rad = 4  # agent  
        t_rad = 10  # target
        # obs_rad_list = self.radius_this  #the radius of obstacle circles


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.length, self.width)
 

            #render the ECDIS---------------------------------------------------------------------
            for i in range(len(binary_grid)):
                for j in range(len(binary_grid[0])):
                    if binary_grid[i][j] == 1:
                        color = (25/255, 101/255, 149/255)  # ocean
                    else:
                        color = (255/255, 228/255,181/255)  # shoreline
                    rect = rendering.FilledPolygon([(j, i),((j + 1), i),((j + 1), (i + 1)), (j, (i + 1))])
                    rect.set_color(*color)
                    self.viewer.add_geom(rect)


            # render the detected circles----------------------------------------------------------------------------------
            numb=0
            global circle_points
            circle_points={}
            global circle_trans
            circle_trans={}
            for number in radii:
                from gym.envs.classic_control import rendering
                key="circle_point"+str(numb)
                circle_points[key]= rendering.make_circle(number, 30, True)
                circle_points[key].set_color(119/255, 136/255, 153/255)
                self.viewer.add_geom(circle_points[key])
                circle_trans[key] = rendering.Transform()
                circle_points[key].add_attr(circle_trans[key])
                numb += 1

            target = rendering.make_circle(t_rad, 30, True)
            target.set_color(255/255, 215/255, 0)
            self.viewer.add_geom(target)
            target_circle = rendering.make_circle(t_rad, 30, False)
            target_circle.set_color(0, 0, 0)
            self.viewer.add_geom(target_circle)
            self.target_trans = rendering.Transform()
            target.add_attr(self.target_trans)
            target_circle.add_attr(self.target_trans)

            self.agent = rendering.make_circle(rad, 30, True)
            self.agent.set_color(0, 1, 0)
            self.viewer.add_geom(self.agent)
            self.agent_trans = rendering.Transform()
            self.agent.add_attr(self.agent_trans)
            agent_circle = rendering.make_circle(rad, 30, False)
            agent_circle.set_color(0, 0, 0)
            agent_circle.add_attr(self.agent_trans)
            self.viewer.add_geom(agent_circle)


        ppx, ppy, angle, tx, ty, u2_right, v2_right, u_right, v_right= self.state
        self.target_trans.set_translation(tx, ty)
        self.agent_trans.set_translation(ppx, ppy)


        # Render obstacles---------------------------------------------------------------------------
        nuum=0
        for k in center_change:
            key="circle_point"+str(nuum)
            circle_trans[key].set_translation(k[0], k[1])
            nuum+=1

        if DRAW == True:

            points2 = [(x[k], y[k]) for k in range(len(x))]
            self.viewer.draw_polyline(points2, color=(0, 0, 255), linewidth=5)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
       



if __name__ == "__main__":
    env = PuckWorldEnv()
    print("hello")
    nfs = env.observation_space.shape[0]
    nfa = env.action_space.n
    print("nfs:%d; nfa:%d" % (nfs,nfa))
    print(env.observation_space)
    print(env.action_space)
    done = False

    for _ in range(10000):
        # env.__init__()
        env.reset()
        while not done :
            env.render(0,0)
            s,r,done,collision= env.step(env.action_space.sample())
            



    print("env closed")


