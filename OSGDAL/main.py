import numpy as np
import gdal
import ogr
import scipy
from skimage import exposure
from skimage.segmentation import slic
import time
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.simplefilter(action='ignore',
                      category=FutureWarning)

colombia_fn = 'Cropped_Colombia_Area_3.tiff'
segments_fn = \
    'C:/temp/eosImages/segments_final.tif'
driverTiff = gdal.GetDriverByName('GTiff')
colombia_ds = gdal.Open(colombia_fn)
nbands = colombia_ds.RasterCount
band_data = []
print('bands', colombia_ds.RasterCount, 'rows',
      colombia_ds.RasterYSize, 'columns',
      colombia_ds.RasterXSize)
for i in range(1, nbands + 1):
    band = \
        colombia_ds.GetRasterBand(i).ReadAsArray()
    band_data.append(band)
band_data = np.dstack(band_data)
img = exposure.rescale_intensity(band_data)

seg_start = time.time()
segments = slic(img, n_segments=68250,
                compactness=0.1)
print('segments complete', time.time() - seg_start)


def segment_features(segment_pixels):
    features = []
    npixels, nbands = segment_pixels.shape
    for b in range(nbands):
        # noinspection PyUnresolvedReferences
        stats = scipy.stats.describe(segment_pixels[:, b])
        band_stats = list(stats.minmax) + list(stats)[2:]
        if npixels == 1:
            # in this case the variance = nan, change it 0.0
            band_stats[3] = 0.0
        features += band_stats
    return features


obj_start = time.time()
segment_ids = np.unique(segments)  # Assigning IDs to each of the unique segments. 1D array
objects = []
object_ids = []

for id in segment_ids:
    # noinspection PyUnresolvedReferences
    segment_pixels = img[segments == id]
    print('pixels for id', id, segment_pixels.shape)
    object_features = segment_features(segment_pixels)
    objects.append(object_features)
    object_ids.append(id)

print('created', len(objects), 'objects with', len(objects[0]), 'variables in', time.time() - obj_start, 'seconds')
segments_ds = driverTiff.Create(
    segments_fn, colombia_ds.RasterXSize,
    colombia_ds.RasterYSize, 1,
    gdal.GDT_Float32)
segments_ds.SetGeoTransform(
    colombia_ds.GetGeoTransform())
segments_ds.SetProjection(
    colombia_ds.GetProjectionRef())
segments_ds.GetRasterBand(1).WriteArray(segments)
segments_ds = None

train_fn = 'C:/temp/eosImages/train.shp'
train_ds = ogr.Open(train_fn)
lyr = train_ds.GetLayer()
driver = gdal.GetDriverByName('MEM')
target_ds = driver.Create('', colombia_ds.RasterXSize, colombia_ds.RasterYSize, 1, gdal.GDT_UInt16)
target_ds.SetGeoTransform(colombia_ds.GetGeoTransform())
target_ds.SetProjection(colombia_ds.GetProjection())
options = ['ATTRIBUTE=id']
gdal.RasterizeLayer(target_ds, [1], lyr, options=options)

ground_truth = target_ds.GetRasterBand(1).ReadAsArray()

classes = np.unique(ground_truth)[1:]
print('class values', classes)

segments_per_class = {}

for klass in classes:
    segments_of_class = segments[ground_truth == klass]
    segments_per_class[klass] = set(segments_of_class)
    print('training segments for class', klass, ':', len(segments_of_class))

intersection = set()
accum = set()

for class_segments in segments_per_class.values():
    intersection |= accum.intersection(class_segments)
    accum |= class_segments
assert len(intersection) == 0, "Segment(s) represent multiple classes"

train_img = np.copy(segments)
threshold = train_img.max() + 1

for klass in classes:
    class_label = threshold + klass
    for segment_id in segments_per_class[klass]:
        train_img[train_img == segment_id] = class_label

train_img[train_img <= threshold] = 0
train_img[train_img > threshold] -= threshold

training_objects = []
training_labels = []

for klass in classes:
    class_train_object = [v for i, v in enumerate(objects) if segment_ids[i] in segments_per_class[klass]]
    training_labels += [klass] * len(class_train_object)
    training_objects += class_train_object
    print('Training objects for class', klass, ':', len(class_train_object))

classifier = RandomForestClassifier(n_jobs=-1)
classifier.fit(training_objects, training_labels)
print('Fitting Random Forest Classifier')
predicted = classifier.predict(objects)
print('Predicting Classification')

clf = np.copy(segments)
for segment_id, klass in zip(segment_ids, predicted):
    clf[clf == segment_id] = klass

print('Prediction applied to numpy array')

mask = np.sum(img, axis=2)
mask[mask > 0.0] = 1.0
mask[mask == 0] = -1.0
clf = np.multiply(clf, mask)
clf[clf < 0] = -9999.0

print('Saving classification to raster with gdal')

clfds = driverTiff.Create('C:/temp/eosImages/classified.tiff', colombia_ds.RasterXSize, colombia_ds.RasterYSize, 1,
                          gdal.GDT_Float32)
clfds.SetGeoTransform(colombia_ds.GetGeoTransform())
clfds.SetProjection(colombia_ds.GetProjection())
clfds.GetRasterBand(1).SetNoDataValue(-9999.0)
clfds.GetRasterBand(1).WriteArray(clf)
clfds = None

print('Done!')
