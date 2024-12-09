import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw
import random



# load picture------------------------------------
desired_width = 200
desired_length = 300
binary_map_data = np.loadtxt(r'Image processing module\testmap_105108.txt', dtype=int)
one_indices = np.argwhere(binary_map_data == 1)
image = Image.new("RGB", (desired_length, desired_width), "white")
draw = ImageDraw.Draw(image)
bw_image = Image.fromarray((binary_map_data * 255).astype(np.uint8))
image.paste(bw_image, (0, 0))  
filtered_indices = [(y, x) for y, x in one_indices if (0 <= x <= 300) and (0 <= y <= 200)]



# A random position generates a circle with five random radii------------------------------------
for _ in range(10):
    valid_position = False
    diameter = random.randint(4, 6)
    while not valid_position:
        center = random.choice(filtered_indices)   
        if center[0] - diameter // 2 >= 0 and center[0] + diameter // 2 < desired_length \
            and center[1] - diameter // 2 >= 0 and center[1] + diameter // 2 < desired_width:
            if binary_map_data[center[1] - diameter // 2][center[0] - diameter // 2] == 1 \
            and binary_map_data[center[1] + diameter // 2][ center[0] + diameter // 2] == 1 \
            and binary_map_data[center[1] - diameter // 2][ center[0] - diameter // 2] == 1 \
            and binary_map_data[center[1] - diameter // 2][ center[0] + diameter // 2] == 1:
                valid_position = True
            else:
                valid_position = False
        else:
            valid_position = False   
    bounding_box = [
        (center[1] - diameter // 2, center[0] - diameter // 2),
        (center[1] + diameter // 2, center[0] + diameter // 2)
    ]
    draw.ellipse(bounding_box, fill="red")
    print(f"Random Circle: Center {center}, Diameter {diameter}")

# goal point------------------------------------
goal_center = (200-89, 147)
goal_radius = 10
goal_bounding_box = [
    (goal_center[1] - goal_radius, goal_center[0] - goal_radius),
    (goal_center[1] + goal_radius, goal_center[0] + goal_radius)
]
draw.ellipse(goal_bounding_box, fill=(250,109,0))



# agent(from start point)------------------------------------
agent_center = (200-16, 97)
agent_radius = 4
agent_bounding_box = [
    (agent_center[1] - agent_radius, agent_center[0] - agent_radius),
    (agent_center[1] + agent_radius, agent_center[0] + agent_radius)
]
draw.ellipse(agent_bounding_box, fill=(0,255,0))


image.show()