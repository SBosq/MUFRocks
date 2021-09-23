from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sn

style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (16, 7)

df = pd.read_csv("Rock_data_1.csv")

X_var = df[['X', 'Y']].values
y_var = df['id'].values

X_var = StandardScaler().fit(X_var).transform(X_var.astype(float))

X_train, X_test, y_train, y_test = \
    train_test_split(X_var, y_var,
                     test_size=0.3,
                     random_state=0)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_var, clf.predict(X_var))
sn.heatmap(cm, annot=True, fmt='g')

print('F1 Score:', metrics.f1_score(y_test, y_pred, average='macro'))

print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
print()

print("Diagonal: " + str(cm.diagonal()))
print("Matrix Sum: " + str(cm.sum(axis=0)))
print()
accuracy = cm.diagonal() / cm.sum(axis=0)
print("Accuracy: " + str(accuracy) + '\n')
print(classification_report(y_test, y_pred))
print("Precision Score: ", metrics.precision_score(y_test, y_pred, average='macro'))
print("Recall Score: ", metrics.recall_score(y_test, y_pred, average='macro'))
plt.show()
