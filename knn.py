import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics



df = pd.read_csv('teleCust1000t.csv')

x = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
x[0:5]

y = df['custcat'].values
y[0:5]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

scaler = preprocessing.StandardScaler().fit(x_train)

x_train_norm = scaler.transform(x_train.astype(float))
x_train_norm[0:5]

x_test_norm = scaler.transform(x_test.astype(float))
x_test_norm[0:5]

k = 4

neigh = KNeighborsClassifier(n_neighbors = k).fit(x_train_norm, y_train)

print("Train Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(x_train_norm)))
print("Test Accuracy: ", metrics.accuracy_score(y_test, neigh.predict(x_test_norm)))

k = 6

neigh = KNeighborsClassifier(n_neighbors = k).fit(x_train_norm, y_train)

print("Train Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(x_train_norm)))
print("Test Accuracy: ", metrics.accuracy_score(y_test, neigh.predict(x_test_norm)))


