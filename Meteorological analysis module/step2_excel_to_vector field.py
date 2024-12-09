import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

#vector field--wind(u10/v10)---------------------------------------------------------------------------
excel_file_path_u10 = r'Meteorological analysis module\Meterorological Data\u10.xlsx'
excel_file_path_v10 = r'Meteorological analysis module\Meterorological Data\v10.xlsx'
df1 = pd.read_excel(excel_file_path_u10)
df2 = pd.read_excel(excel_file_path_v10)
array1 = df1.values
array2 = df2.values
list_u10 = [[round(num, 3) for num in row] for row in array1]
list_v10 = [[round(num, 3) for num in row] for row in array2]
u10_array = np.array(list_u10)
v10_array = np.array(list_v10)
u10_array_flipped = np.flipud(u10_array)
v10_array_flipped = np.flipud(v10_array)
magnitude = np.sqrt(u10_array_flipped**2 + v10_array_flipped**2)
x, y = np.meshgrid(np.arange(u10_array_flipped.shape[1]), np.arange(u10_array_flipped.shape[0]))
plt.figure(figsize=(14, 6))
plt.quiver(x, y, u10_array_flipped, v10_array_flipped, magnitude, scale=50, cmap='viridis')
plt.xlim(0, u10_array_flipped.shape[1])
plt.ylim(0, u10_array_flipped.shape[0])
plt.xticks(np.arange(0, u10_array_flipped.shape[1], 2.5))
plt.yticks(np.arange(0, u10_array_flipped.shape[0], 2.5))
plt.colorbar()
plt.title('Wind Vector (10m) Field with Magnitude')
plt.xlabel('Longitude (°E)')
plt.ylabel('Latitude (°S)')
plt.savefig(r'Meteorological analysis module\Meterorological Data\wind_vector_10m_field.png', bbox_inches='tight')



#vector field--wind(u2/v2)----------------------------------------------------------------------------------
excel_file_path_u2 = r'Meteorological analysis module\Meterorological Data\u2.xlsx'
excel_file_path_v2 = r'Meteorological analysis module\Meterorological Data\v2.xlsx'
df_u2 = pd.read_excel(excel_file_path_u2)
df_v2 = pd.read_excel(excel_file_path_v2)
array_u2 = df_u2.values
array_v2 = df_v2.values
list_u2 = [[round(num, 3) for num in row] for row in array_u2]
list_v2 = [[round(num, 3) for num in row] for row in array_v2]
u2_array = np.array(list_u2)
v2_array = np.array(list_v2)
u2_array_flipped = np.flipud(u2_array)
v2_array_flipped = np.flipud(v2_array)
magnitude = np.sqrt(u2_array_flipped**2 + v2_array_flipped**2)
x, y = np.meshgrid(np.arange(u2_array_flipped.shape[1]), np.arange(u2_array_flipped.shape[0]))
plt.figure(figsize=(14, 6))
plt.quiver(x, y, u2_array_flipped, v2_array_flipped, magnitude, scale=50, cmap='viridis')
plt.colorbar()
plt.title(' Wind Vector(2m) Field with Magnitude')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.xticks(np.arange(0, u2_array_flipped.shape[1], 2.5))
plt.yticks(np.arange(0, u2_array_flipped.shape[0], 2.5))
plt.savefig(r'Meteorological analysis module\Meterorological Data\wind_vector_2m_field.png', bbox_inches='tight')


#vector field--ocean current field(u/v)----------------------------------------------------------------------------
excel_file_path_u = r'Meteorological analysis module\Meterorological Data\u.xlsx'
excel_file_path_v = r'Meteorological analysis module\Meterorological Data\v.xlsx'
df_u = pd.read_excel(excel_file_path_u)
df_v = pd.read_excel(excel_file_path_v)
array_u = df_u.values
array_v = df_v.values
list_u = [[round(num, 3) for num in row] for row in array_u]
list_v = [[round(num, 3) for num in row] for row in array_v]
u_array = np.array(list_u)
v_array = np.array(list_v)
u_array_flipped = np.flipud(u_array)
v_array_flipped = np.flipud(v_array)
magnitude = np.sqrt(u_array_flipped**2 + v_array_flipped**2)
x, y = np.meshgrid(np.arange(u_array_flipped.shape[1]), np.arange(u_array_flipped.shape[0]))
plt.figure(figsize=(14, 6))
plt.quiver(x, y, u_array_flipped, v_array_flipped, magnitude, scale=1, cmap='viridis')
plt.colorbar()
plt.title('Ocean Current Vector(-2.5m) Field with Magnitude')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.xticks(np.arange(0, u_array_flipped.shape[1], 2.5))
plt.yticks(np.arange(0, u_array_flipped.shape[0], 2.5))
plt.savefig(r'文件语义信息_矢量场\ocean_current_vector_field.png', bbox_inches='tight')

