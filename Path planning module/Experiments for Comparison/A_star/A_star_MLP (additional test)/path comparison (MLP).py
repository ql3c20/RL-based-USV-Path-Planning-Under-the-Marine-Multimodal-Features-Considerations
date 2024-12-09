from scipy import ndimage
from PIL import Image, ImageDraw, ImageColor, ImageFont
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev





fig, ax = plt.subplots(figsize=(3, 3))
binary_map_data = np.loadtxt(r'Image processing module\Geographic Data\testmap_105108.txt', dtype=int)
binary_map_data = np.array(binary_map_data)
binary_array_revers = np.flipud(binary_map_data)
ax.set_aspect('equal') 
ax.set_xlim(0, binary_array_revers.shape[1])  
ax.set_ylim(0, binary_array_revers.shape[0])  
color_map = {
    0: [255, 228, 181],  # rgb(255, 228, 181)
    1: [25, 101, 149]   # rgb(135, 206, 250)
}
cmap = plt.cm.colors.ListedColormap([[255/255, 228/255, 181/255], [25/255, 101/255, 149/255]])
ax.imshow(binary_array_revers, cmap=cmap, interpolation='none', aspect='auto', vmin=0, vmax=1)



# load the vector data of wind field
excel_file_path_u2 = r'Meteorological analysis module\Meterorological Data\u2_new.xlsx'
excel_file_path_v2 = r'Meteorological analysis module\Meterorological Data\v2_new.xlsx'
df_u2 = pd.read_excel(excel_file_path_u2)
df_v2 = pd.read_excel(excel_file_path_v2)
array_u2 = df_u2.iloc[4:16, 12:24].values
array_v2 = df_v2.iloc[4:16, 12:24].values
list_u2 = [[round(num, 5) for num in row] for row in array_u2]
list_v2 = [[round(num, 5) for num in row] for row in array_v2]
u2_array = np.array(list_u2)
v2_array = np.array(list_v2)
u2_array_flipped = np.flipud(u2_array)
v2_array_flipped = np.flipud(v2_array)
for i in range(0, 12):
    for j in range(0, 12):
        u_value = u2_array_flipped[i, j]
        v_value = v2_array_flipped[i, j]
        if binary_array_revers[int(i / 12 * 300), int(j / 12 * 300)] == 1:
            start_point = (j / 12 * 300, i / 12 * 300)
            ax.quiver(start_point[0], start_point[1], 60*u_value, 60*v_value, color=(245 / 255, 245 / 255, 220 / 255), angles='xy', scale_units='xy', scale=4, width=0.008)


# load the vector data of ocean current field
excel_file_path_u = r'Meteorological analysis module\Meterorological Data\u_interpolated.xlsx'
excel_file_path_v = r'Meteorological analysis module\Meterorological Data\v_interpolated.xlsx'
df_u = pd.read_excel(excel_file_path_u)
df_v = pd.read_excel(excel_file_path_v)
array_u = df_u.iloc[4:16, 12:24].values
array_v = df_v.iloc[4:16, 12:24].values
list_u = [[round(num, 5) for num in row] for row in array_u]
list_v = [[round(num, 5) for num in row] for row in array_v]
u_array = np.array(list_u)
v_array = np.array(list_v)
u_array_flipped = np.flipud(u_array)
v_array_flipped = np.flipud(v_array)
for i in range(0, 12):
    for j in range(0, 12):
        u = u_array_flipped[i, j]
        v = v_array_flipped[i, j]
        if binary_array_revers[int(i / 12 * 300), int(j / 12 * 300)] == 1:
            start_point = (j / 12 * 300, i / 12 * 300)
            ax.quiver(start_point[0], start_point[1], 1200*u, 1200*v, color=(135 / 255, 206 / 255, 250 / 255), angles='xy', scale_units='xy', scale=4, width=0.008)


# fusion
excel_file1 = r'Path planning module\Experiments for Comparison\fusionDQN_x9522_y14992.xlsx'  # add your trainning path
df1 = pd.read_excel(excel_file1)
x1 = df1.iloc[:, 0]
y1 = df1.iloc[:, 1]
angle = df1.iloc[:, 2]
tck, u = splprep([x1, y1], s=0)
u_new = np.linspace(0, 1, 1000)
xy_smooth = splev(u_new, tck)
ax.plot(xy_smooth[0], xy_smooth[1], color=(255/255,0/255,0/255), linewidth=2)



# DQN
excel_file2 = r'Path planning module\Experiments for Comparison\DQN_x9522_y14992.xlsx'  # add your trainning file path
df2 = pd.read_excel(excel_file2)
x2 = df2.iloc[:, 4]
y2 = df2.iloc[:, 5]
angle = df2.iloc[:, 6]
tck, u = splprep([x2, y2], s=0)
u_new = np.linspace(0, 1, 1000)
xy_smooth = splev(u_new, tck)
ax.plot(xy_smooth[0], xy_smooth[1], color=(235/255,235/255,235/255), linewidth=2)


# Astar 
excel_file_3 = r'Path planning module\Experiments for Comparison\Astar_x9522_y14992.xlsx'  # add your trainning file path
df_3 = pd.read_excel(excel_file_3)
x_3 = df_3.iloc[:, 0]
y_3 = df_3.iloc[:, 1]
tck_3, u_3 = splprep([x_3, y_3], s=0)
u_3_new = np.linspace(0, 1, 1000)
xy_3_smooth = splev(u_3_new, tck_3)
ax.plot(xy_3_smooth[0], xy_3_smooth[1], color=(185/255,185/255,185/255), linewidth=2)

# Astar_MLP
excel_file_4 = r'Path planning module\Experiments for Comparison\A_star\A_star_MLP (additional test)\Astar_MLP_test.xlsx'  # add your trainning file path
df_4 = pd.read_excel(excel_file_4)
x_4 = df_4.iloc[:, 0]
y_4 = df_4.iloc[:, 1]
tck_4, u_4 = splprep([x_4, y_4], s=0)
u_4_new = np.linspace(0, 1, 1000)
xy_4_smooth = splev(u_4_new, tck_4)
ax.plot(xy_4_smooth[0], xy_4_smooth[1], color=(190/255,220/255,170/255), linewidth=2)




# obstacles
obstacle_centers = [(26, 175), (122, 102), (140, 161), (129, 56), (106, 10), (215, 154), (200, 16), (105, 33)]
radius = [2, 4, 2, 3, 2, 2, 4, 3]
for center, r in zip(obstacle_centers, radius):
    ax.add_patch(plt.Circle((center[0], center[1]), r, color=(212 / 255, 213 / 255, 214 / 255)))

# agent
agent_center = (95, 22)
agent_radius = 2
ax.add_patch(plt.Circle((agent_center[0], agent_center[1]), agent_radius, color=(0, 1, 0), zorder= 10))


# goal point
goal_center = (147, 88)
goal_radius = 10
ax.add_patch(plt.Circle((goal_center[0], goal_center[1]), goal_radius, color=(250 / 255, 109 / 255, 0),zorder= 9))
plt.axis('off')  


left, right = 76, 174
bottom, top = 10, 108
ax.set_xlim(left, right)
ax.set_ylim(bottom, top)
# 在外边界添加黑色边框
ax.add_patch(plt.Rectangle((left, bottom),
                           right-left, top-bottom,
                           fill=False, edgecolor='black', linewidth=5))
plt.savefig(r'Path planning module\Experiments for Comparison\A_star\A_star_MLP (additional test)\MLP_comparison.eps')
plt.savefig(r'Path planning module\Experiments for Comparison\A_star\A_star_MLP (additional test)\MLP_comparison.pdf')
plt.savefig(r'Path planning module\Experiments for Comparison\A_star\A_star_MLP (additional test)\MLP_comparison.png')

plt.show()


