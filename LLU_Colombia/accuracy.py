import numpy as np
import gdal
import ogr
from sklearn import metrics
from sklearn.metrics import classification_report


naip_fn = 'Red_Colombia_Area_3.tiff'
driverTiff = gdal.GetDriverByName('GTiff')
naip_ds = gdal.Open(naip_fn)

test_fn = 'C:/temp/eosImages/red_Test.shp'
test_ds = ogr.Open(test_fn)
lyr = test_ds.GetLayer()
driver = gdal.GetDriverByName('MEM')
target_ds = driver.Create('', naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_UInt16)
target_ds.SetGeoTransform(naip_ds.GetGeoTransform())
target_ds.SetProjection(naip_ds.GetProjection())
options = ['ATTRIBUTE=id']
gdal.RasterizeLayer(target_ds, [1], lyr, options=options)

truth = target_ds.GetRasterBand(1).ReadAsArray()

pred_ds = gdal.Open('C:/temp/eosImages/red_Classified.tif')
pred = pred_ds.GetRasterBand(1).ReadAsArray()

idx = np.nonzero(truth)

cm = metrics.confusion_matrix(truth[idx], pred[idx])

# pixel accuracy
print(cm)
print()

print("Diagonal: " + str(cm.diagonal()))
print("Matrix Sum: " + str(cm.sum(axis=0)))
print()

accuracy = cm.diagonal() / cm.sum(axis=0)
print("Accuracy: " + str(accuracy) + '\n')
print(classification_report(truth[idx], pred[idx]))
