import gdal
import ogr

naip_fn = 'Red_Colombia_Area_3.tiff'
naip_ds = gdal.Open(naip_fn)

train_fn = 'C:/temp/eosImages/red_Train.shp'
train_ds = ogr.Open(train_fn)
lyr = train_ds.GetLayer()
driver = gdal.GetDriverByName('MEM')
target_ds = driver.Create('', naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_UInt16)
target_ds.SetGeoTransform(naip_ds.GetGeoTransform())
target_ds.SetProjection(naip_ds.GetProjection())
options = ['ATTRIBUTE=id']
gdal.RasterizeLayer(target_ds, [1], lyr, options=options)
data = target_ds.GetRasterBand(1).ReadAsArray()
print('min', data.min(), 'max', data.max(), 'mean', data.mean())
