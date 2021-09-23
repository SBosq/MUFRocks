import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

df = pd.read_csv("Rock_data_1.csv")

X = df[['X', 'Y']].values  # , 'RockTypes'
y = df['id'].values

X_var = StandardScaler().fit(X).transform(X.astype(float))
X_train, X_test, y_train, y_test = \
    train_test_split(X, y,
                     random_state=None,
                     test_size=0.3)
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y, clf.predict(X))
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print(cm)
print()

print("Diagonal: " + str(cm.diagonal()))
print("Matrix Sum: " + str(cm.sum(axis=0)))
print()

accuracy = cm.diagonal() / cm.sum(axis=0)
print("Accuracy: " + str(accuracy) + '\n')
print(classification_report(y_test, y_pred))
# print("Precision: ", metrics.precision_score(y_test, y_pred))
# print("Recall: ", metrics.recall_score(y_test, y_pred))

"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report

df = pd.read_csv("Rock_data_1.csv")
df1 = pd.read_csv("test.csv")
df2 = pd.read_csv("train.csv")

X = df[['X', 'Y']].values
# X = df1[['X', 'Y']].values
# X = df2[['X', 'Y']].values
y = df['id'].values
# y = df1['id'].values
# y = df2['id'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=None, test_size=0.4)
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
predicted_values = clf.predict(X_train)
print("Accuracy: ", metrics.accuracy_score(y_test, predicted_values))
print(classification_report(y_test, predicted_values))
# print("Precision: ", metrics.precision_score(y_test, y_pred))
# print("Recall: ", metrics.recall_score(y_test, y_pred))"""
