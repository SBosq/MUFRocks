import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

df = pd.read_csv("red_Class.csv")
X = df[['X', 'Y']].values
y = df['id'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=None, test_size=0.3)
model = LogisticRegression(solver='liblinear', random_state=0).fit(X, y)
# print(model.predict(X))
y_pred = model.predict(X_test)
print("Model Accuracy:", model.score(X, y))
print()
# print(confusion_matrix(y, model.predict(X)))
cm = confusion_matrix(y, model.predict(X))
print(cm)
print()

print("Diagonal: " + str(cm.diagonal()))
print("Matrix Sum: " + str(cm.sum(axis=0)))
print()

accuracy = cm.diagonal() / cm.sum(axis=0)
print("Accuracy: " + str(accuracy) + '\n')
print(classification_report(y_test, y_pred))
