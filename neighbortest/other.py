from termcolor import colored as cl
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sn

style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (16, 7)

df = pd.read_csv("Rock_data_1.csv")
print(cl(df, attrs=['bold']))

X_var = df[['X', 'Y']].values
y_var = df['id'].values

print(cl('X variable :', attrs=['bold']), X_var[: 5])
print(cl('Y variable :', attrs=['bold']), y_var[: 5])

X_var = StandardScaler().fit(X_var).transform(X_var.astype(float))
print(cl(X_var[: 5], attrs=['bold']))

X_train, X_test, y_train, y_test = \
    train_test_split(X_var, y_var,
                     test_size=0.3,
                     random_state=0)

print(cl('Train set shape :', attrs=['bold']), X_train.shape, y_train.shape)
print(cl('Test set shape :', attrs=['bold']), X_test.shape, y_test.shape)

k = 4
neigh = KNeighborsClassifier(n_neighbors=k)
neigh.fit(X_train, y_train)

print(cl(neigh, attrs=['bold']))

yhat = neigh.predict(X_test)

print(cl('Prediction Accuracy Score (%) :', attrs=['bold']), round(accuracy_score(y_test, yhat) * 100, 2))
print()
cm = confusion_matrix(y_var, neigh.predict(X_var))
sn.heatmap(cm, annot=True, fmt='g')
print(cm)
print()

print("Diagonal: " + str(cm.diagonal()))
print("Matrix Sum: " + str(cm.sum(axis=0)))
print()

accuracy = cm.diagonal() / cm.sum(axis=0)
print("Accuracy: " + str(accuracy) + '\n')
print(classification_report(y_test, yhat))
print("Precision Score: ", metrics.precision_score(y_test, yhat, average='macro'))
print("Recall Score: ", metrics.recall_score(y_test, yhat, average='macro'))
plt.show()
