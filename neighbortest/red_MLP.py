import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv("red_Class.csv")

le = preprocessing.LabelEncoder()

X = df[['X', 'Y']].values
y = df['id'].values
print(df.head())
training_a, testing_a, training_b, testing_b = train_test_split(X, y, test_size=0.3)
myscaler = StandardScaler()
myscaler.fit(training_a)
training_a = myscaler.transform(training_a)
testing_a = myscaler.transform(testing_a)
m1 = MLPClassifier(hidden_layer_sizes=(12, 13, 14), activation='relu', solver='adam', max_iter=2500)
m1.fit(training_a, training_b)
predicted_values = m1.predict(testing_a)
print("Model Accuracy:", m1.score(X, y))
print(confusion_matrix(testing_b, predicted_values))
cm = confusion_matrix(testing_b, predicted_values)
accuracy = cm.diagonal() / cm.sum(axis=0)
print("\nAccuracy: " + str(accuracy) + '\n')
print(classification_report(testing_b, predicted_values))

"""X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training and 30% test
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(4, 2), random_state=0, verbose=True, learning_rate_init=0.01)
clf.fit(X_train, y_train)
ypred = clf.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, ypred))
"""
