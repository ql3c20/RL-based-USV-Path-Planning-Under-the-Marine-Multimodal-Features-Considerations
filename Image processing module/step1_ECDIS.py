import os
import numpy as np
import matplotlib.ticker as mticker
import matplotlib as mpl
import pickle
import cartopy.crs as ccrs
import geopandas as gpd
import shapefile
from rasterio.features import rasterize
import rasterio


from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import pyplot as plt
from shapely.geometry import shape, Polygon

colour_dic = {'1': "white", '2': "black", '3': "red", '4': "green", '5': "blue", '6': "yellow", '7': "grey",
              '8': "brown", '9': "amber", '10': "violet", '11': "orange", '12': "magenta", '13': "pink"}
colour_luminance_dic = {'0': "black", '1': "darkgray", '2': "mediumgray", '3': "lightgray", '4': "palegray", '5': "white"}
pattern_dic = {'1': "-----", '2': "|||||", '3': r"\\" + "\\", '4': "xxxxx", '5': ".....", '6': "-----+"}
boyshp_dic = {'1': "^", '2': "v", '3': "o", '4': "|", '5': "s", '6': "D", '7': "P", '8': "X"}


mpl.rcParams["font.family"] = 'Arial'  
mpl.rcParams["mathtext.fontset"] = 'cm'  
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.linewidth"] = 1


class map_enc_charts:
    def __init__(self, region, pixel_size, folder_path):
        self.region = region  
        self.pixel_size = pixel_size  
        self.folder_path = folder_path  

    def enc_2d_init(self):
        shape_reader_files, gpd_read_files = self.enc_file_layer()
        self.enc_obstacle_areas_analysis_layer(gpd_read_files)
        # surfaces_dict, features_dict = self.enc_data_analysis_layer(shape_reader_files)
        # ax = self.enc_render_layer(surfaces_dict, features_dict)
        # obstacle_mask, geo_polygons = self.enc_obstacle_areas_analysis_layer(gpd_read_files)
        # return ax, obstacle_mask, geo_polygons

    def enc_file_layer(self):
        shape_reader_files = []
        gpd_read_files = []
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith('.shp'):
                file_path = os.path.join(self.folder_path, file_name)
                shape_reader_files.append(shapefile.Reader(file_path))
                gpd_read_files.append(gpd.read_file(file_path))

        return shape_reader_files, gpd_read_files

    @staticmethod
    def enc_data_analysis_layer(shape_reader_files):
        surface_dict = {}  
        features_dict = {}  

        for shape_reader_file in shape_reader_files:
            field_names = [field[0] for field in shape_reader_file.fields[1:]]
            for shape_record in shape_reader_file.iterShapeRecords():
                if shape_record.shape.shapeType == 0:
                    continue
                shape_geom = shape(shape_record.shape.__geo_interface__)
                if shape_geom.is_valid:
                    attributes = dict(zip(field_names, shape_record.record))
                    feature = (shape_geom, attributes)
                    grup = attributes.get('GRUP')
                    layer = attributes.get('OBJL')
                    if grup == 1:
                        if layer not in surface_dict:
                            surface_dict[layer] = []
                        surface_dict[layer].append(feature)
                    elif grup == 2:
                        if layer not in features_dict:
                            features_dict[layer] = {}
                        feature_type = attributes.get('OBJNAM', 'unknown')
                        if feature_type not in features_dict[layer]:
                            features_dict[layer][feature_type] = []
                        features_dict[layer][feature_type].append(feature)

        return surface_dict, features_dict

    def enc_render_layer(self, surfaces_dict, features_dict):
        ax = plt.axes(projection=ccrs.PlateCarree())
        for surface_layer in surfaces_dict:
            for shape_geom, attributes in surfaces_dict[surface_layer]:
                if shape_geom.geom_type == "Polygon":
                    if surface_layer == 71:
                        ax.add_geometries([shape_geom], crs=ccrs.PlateCarree(), edgecolor="black", alpha=0.8,
                                          facecolor='navajowhite')
                    else:
                        ax.add_geometries([shape_geom], crs=ccrs.PlateCarree(), edgecolor="black", alpha=0.5,
                                          facecolor='lightskyblue')
                else:
                    print("contains other surface info, but do not figure. please optimize the code!")
                    return

        for features_layer in features_dict:
            for features in features_dict[features_layer]:
                for shape_geom, attributes in features_dict[features_layer][features]:
                    color, luminance, pattern = self.get_colour(attributes.get("COLOUR", ""),
                                                                attributes.get("COLPAT", ""))
                    if shape_geom.geom_type == "Polygon":
                        ax.add_geometries([shape_geom], crs=ccrs.PlateCarree(),
                                          edgecolor="black", facecolor=color, alpha=0.5)
                    elif shape_geom.geom_type == "LineString":
                        ax.add_geometries([shape_geom], crs=ccrs.PlateCarree(),
                                          edgecolor=color, facecolor="None", alpha=1)
                    elif shape_geom.geom_type == "Point":
                        marker = attributes.get('BOYSHP', "")
                        if marker != "":
                            marker = boyshp_dic[str(marker)]
                        else:
                            marker = "o"
                        point = shape_geom.coords[0]
                        ax.plot(point[0], point[1], marker=marker, markersize=3, c=color, alpha=1,
                                transform=ccrs.PlateCarree())



        # -----------Add latitude and longitude---------------------------------------
        ax.coastlines(resolution='50m') 

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.8, color='k', alpha=0.3, linestyle='--')
        gl.top_labels = False 
        gl.right_labels = False  
        gl.xformatter = LONGITUDE_FORMATTER  
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlocator = mticker.FixedLocator(np.arange(self.region[0] - 0.5, self.region[1] + 0.5, 1))
        gl.ylocator = mticker.FixedLocator(np.arange(self.region[2] - 0.5, self.region[3] + 0.5, 1))
        gl.xlabel_style = {'size': 9}
        gl.ylabel_style = {'size': 9}
        ax.set_extent(self.region)  
        ax.patch.set_facecolor('#f0f0f0') 
        # plt.show()

        return ax

    @staticmethod
    def get_colour(colour, pattern):
        color, luminance, patt = "None", "lightgray", "....."
        if len(colour) == 1:
            color = colour_dic[colour[0]]
        elif len(colour) > 1:
            colour = colour.split(',')
            color = colour_dic[colour[0]]
            luminance = colour_luminance_dic[colour[1]]

        if pattern != '':
            patt = pattern_dic[pattern]

        return color, luminance, patt

    # +++++++++++++++++++++++++  build env including obstacle and bound  +++++++++++++++++++++++++++++++++++
    def enc_obstacle_areas_analysis_layer(self, gpd_read_files):
        map_height = int((self.region[3] - self.region[2]) / self.pixel_size)  
        map_width = int((self.region[1] - self.region[0]) / self.pixel_size)
        mask = np.zeros([map_width, map_height], dtype=np.uint8) 

        # Rasterization is the process of converting vectors (i.e., points, lines, and surfaces) into rasters. A point will become the center of the pixel; A line will fill the width with pixel values, while the polygon will be drawn entirely in pixels.
        polygons = []
        max_polygon = None
        for read_file in gpd_read_files:
            if read_file.geometry.type[0] not in ['Polygon', 'Point']:
                continue
            for index, row in read_file[read_file.geometry.type == 'Polygon'].iterrows():
                if row['GRUP'] != 1 or row['OBJL'] == 42:
                    continue



                polygons.append(row['geometry'])

            for index, row in read_file[read_file.geometry.type == 'Point'].iterrows():
                obj_name = row.get("OBJNAM", "")
                if row['GRUP'] != 2 or obj_name is None or obj_name == "":
                    continue
                lat, lon = int((row['geometry'].y - self.region[2]) / self.pixel_size), \
                           int((row['geometry'].x - self.region[0]) / self.pixel_size)
                try:
                    mask[lat, lon] = True
                except IndexError:
                    pass


        # The polygon geometry object collection is converted to raster data, and the mask two-dimensional array is output
        shapes = [(geom, 1) for geom in polygons]
        mask = rasterize(shapes, out_shape=(map_height, map_width),
                         transform=rasterio.transform.from_bounds(self.region[0], self.region[3], self.region[1],
                                                                  self.region[2], map_width, map_height))
        mask = mask[::-1, :]  


        # print(f"Binary grid data saved to {OUTPUT_FILE_PATH}")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(mask, cmap='binary')
        plt.show()

        return mask, polygons


def main():
    
    my_instance = map_enc_charts([105,108,-4,-1], 0.01, "D:\Desktop\China_South_Sea\China_South_Sea")
    #There is no permission to disclose this file, so some test data will be used in subsequent experiments(Results may be biased)
    #Can be replaced with electronic charts for different regions
    my_instance.enc_2d_init()
    my_instance.enc_file_layer()
    my_instance.enc_obstacle_areas_analysis_layer()

if __name__ == "__main__":
    main()