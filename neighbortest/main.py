from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import gdal
import ogr

ds = "Rock_data_1.csv"

fields = []
rows = []

with open(ds, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)
    print("Total number of rows: %d" % csvreader.line_num)

print('Field names are:' + ', '.join(field for field in fields))
print('\nFirst 5 rows are:\n')
for row in rows[:5]:
    for col in row:
        print("%10s" % col),
    print('\n')

df = pd.read_csv("Rock_data_1.csv")
# df1 = pd.read_csv("test.csv")

X = df[['X', 'Y']].values
y = df['id'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=None, test_size=0.3)

neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()

knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(X_train, y_train)

print(knn.predict(X_test))

acc = knn.score(X_test, y_test)

print()

print('Accuracy: ' + str(float("%.4f" % acc)))
