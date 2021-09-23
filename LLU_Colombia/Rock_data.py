import numpy as np
import geopandas as gpd
import pandas as pd

gdf = gpd.read_file('C:/Users/saulo/Documents/Rock_data.shp')
class_names = gdf['RockTypes'].unique()
print('class names', class_names)
class_ids = np.arange(class_names.size) + 1
print('class ids', class_ids)
df = pd.DataFrame({'RockTypes': class_names, 'id': class_ids})
df.to_csv('C:/temp/eosImages/RType_lookup.csv')
print('gdf without ids', gdf.head())
gdf['id'] = gdf['RockTypes'].map(dict(zip(class_names, class_ids)))
print('gdf with ids', gdf.head())

gdf_train = gdf.sample(frac=0.7)
gdf_test = gdf.drop(gdf_train.index)
print('gdf shape', gdf.shape, 'training shape', gdf_train.shape, 'test', gdf_test.shape)
gdf_train.to_file('C:/temp/eosImages/red_Train.shp')
gdf_test.to_file('C:/temp/eosImages/red_Test.shp')
