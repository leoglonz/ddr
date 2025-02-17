import geopandas as gpd
import pandas as pd

path_1 = "/projects/mhpi/data/MERIT/raw/basins/cat_pfaf_73_MERIT_Hydro_v07_Basins_v01_bugfix1.shp"
path_2 = "/projects/mhpi/data/hydrofabric/v2.2/jrb_2.gpkg"

gdf1 = gpd.read_file(path_1).set_crs(epsg=4326).to_crs(epsg=5070)
gdf2 = gpd.read_file(path_2, layer="divides").to_crs(epsg=5070)

gdf1['gdf1_orig_area'] = gdf1.geometry.area
gdf2['gdf2_orig_area'] = gdf2.geometry.area

# Perform overlay intersection
# This will create new geometries where polygons overlap
intersection = gpd.overlay(gdf1, gdf2, how='intersection')
intersection['intersection_area'] = intersection.geometry.area
intersection['gdf1_pct'] = (intersection['intersection_area'] / intersection['gdf1_orig_area'])

weight_matrix = pd.pivot_table(intersection, 
                             values='gdf1_pct',
                             index='COMID',  # replace with your actual column name from gdf2
                             columns='divide_id',  # replace with your actual column name from gdf1
                             fill_value=0)

weight_matrix.to_csv("/projects/mhpi/data/hydrofabric/v2.2/jrb_transition_matrix.csv")
print("Created transition matrix @ /projects/mhpi/data/hydrofabric/v2.2/jrb_transition_matrix.csv")
