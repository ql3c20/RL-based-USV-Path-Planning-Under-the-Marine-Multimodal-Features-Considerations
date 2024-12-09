import netCDF4 as nc
import pandas as pd
import numpy as np
import math
from scipy.interpolate import RectBivariateSpline


# ERA5——u10v10----------------------------------------
#ERA5 monthly averaged data on single levels from 1940 to present.Accessed: 2023. [Online]. 
# Available: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview
nc_file_path = r'adaptor.mars.internal-202111.nc'
nc_dataset = nc.Dataset(nc_file_path,encoder="gbk")
data1 = nc_dataset.variables['u10'][:]
data2 = nc_dataset.variables['v10'][:]
data3 = data1[0, :, :]
data4 = data2[0, :, :]
df1 = pd.DataFrame(data3)
df2 = pd.DataFrame(data4)
excel_file_path_u10 = r'Meteorological analysis module\Meterorological Data\u10_new.xlsx'
df1.to_excel(excel_file_path_u10, index=False)
excel_file_path_v10 = r'Meteorological analysis module\Meterorological Data\v10_new.xlsx'
df2.to_excel(excel_file_path_v10, index=False)

print(nc_dataset.variables)


# ERA5——u2v2----------------------------------------
#ERA5 monthly averaged data on single levels from 1940 to present.Accessed: 2023. [Online]. 
# Available: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview
nc_dataset_uv = nc.Dataset(nc_file_path, encoder="gbk")
data1 = nc_dataset_uv.variables['u10'][:]
data2 = nc_dataset_uv.variables['v10'][:]
data3 = data1[0, :, :]
data4 = data2[0, :, :]
u2_data = data3*(np.log(2/0.003) / np.log(10/0.003))
v2_data = data4*(np.log(2/0.003) / np.log(10/0.003))
df_u2 = pd.DataFrame(u2_data)
df_v2 = pd.DataFrame(v2_data)
excel_file_path_u2 = r'Meteorological analysis module\Meterorological Data\u2_new.xlsx'
excel_file_path_v2 = r'Meteorological analysis module\Meterorological Data\v2_new.xlsx'
df_u2.to_excel(excel_file_path_u2, index=False)
df_v2.to_excel(excel_file_path_v2, index=False)



# National Marine Science Data Center dataset-------------------------------------------
# Reanalysis CORAv1.0. Accessed: 2023. [Online]. Available: https://mds.nmdis.org.cn/pages/dataViewDetail.html?dataSetId=83
nc_file_path1 = r'chinasea_202111_c.nc'
nc_dataset = nc.Dataset(nc_file_path1,encoder="gbk")
data_u = nc_dataset.variables['u'][:]
data_v = nc_dataset.variables['v'][:]
data_u_right = data_u[0, 5:24, 10:21]
data_v_right = data_v[0, 5:24, 10:21]
u = pd.DataFrame(data_u_right)
v = pd.DataFrame(data_v_right)
excel_file_path_u = r'Meteorological analysis module\Meterorological Data\u.xlsx'
u.to_excel(excel_file_path_u, index=False)
excel_file_path_v = r'Meteorological analysis module\Meterorological Data\v.xlsx'
v.to_excel(excel_file_path_v, index=False)
df_u = pd.read_excel(excel_file_path_u)
df_u = df_u.fillna(0)
df_u = df_u.transpose()
df_u = df_u.iloc[::-1] 
df_u.to_excel(excel_file_path_u, index=False)
df_v = pd.read_excel(excel_file_path_v)
df_v = df_v.fillna(0)
df_v = df_v.transpose()
df_v = df_v.iloc[::-1]
df_v.to_excel(excel_file_path_v, index=False)

# Bilinear interpolation(Multiple difference methods can be substituted)
old_rows, old_cols = df_u.shape
spline_u = RectBivariateSpline(np.arange(old_rows), np.arange(old_cols), df_u.values)
spline_v = RectBivariateSpline(np.arange(old_rows), np.arange(old_cols), df_v.values)
new_rows, new_cols = 21, 37
new_row_coords = np.linspace(0, old_rows - 1, new_rows)
new_col_coords = np.linspace(0, old_cols - 1, new_cols)
new_data_u = spline_u(new_row_coords, new_col_coords)
new_data_v = spline_v(new_row_coords, new_col_coords)
new_df_u = pd.DataFrame(new_data_u)
new_df_v = pd.DataFrame(new_data_v)
new_excel_file_path_u = r'Meteorological analysis module\Meterorological Data\u_interpolated.xlsx'
new_excel_file_path_v = r'Meteorological analysis module\Meterorological Data\v_interpolated.xlsx'
new_df_u.to_excel(new_excel_file_path_u, index=False)
new_df_v.to_excel(new_excel_file_path_v, index=False)





