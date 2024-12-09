import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# Read the binary map----------------------------------
binary_map_data = np.loadtxt(r'Image processing module\Geographic Data\testmap_original_105108.txt', dtype=int)
non_one_indices = np.nonzero(binary_map_data != 1)
min_y, max_y, min_x, max_x = np.min(non_one_indices[0]), np.max(non_one_indices[0]), np.min(non_one_indices[1]), np.max(non_one_indices[1])
cropped_map = binary_map_data[min_y:max_y+1, min_x:max_x+1]
non_zero_indices = np.nonzero(cropped_map != 0)
min_y, max_y, min_x, max_x = np.min(non_zero_indices[0]), np.max(non_zero_indices[0]), np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
cropped_map = cropped_map[min_y:max_y+1, min_x:max_x+1]




#Crop picture to 300*300-------------------------------------------------
desired_width = 300
desired_length = 300
scale_width = desired_width / cropped_map.shape[0]
scale_length = desired_length / cropped_map.shape[1]
scaled_map = ndimage.zoom(cropped_map, (scale_width, scale_length), order=1)
scaled_map = np.round(scaled_map).astype(int)
np.savetxt(r'Image processing module\Geographic Data\testmap_105108.txt', scaled_map, fmt='%d')



binary_map_data = np.loadtxt(r'Image processing module\Geographic Data\testmap_105108.txt', dtype=int)
# print("Shape of binary_map_data:", binary_map_data.shape)
image = Image.fromarray((binary_map_data * 255).astype(np.uint8))  # Converts binary data to pixel values of 0-255

# image.show()
plt.imshow(image, cmap='gray')  
plt.show()




