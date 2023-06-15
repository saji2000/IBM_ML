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
print(y[0:5])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

scaler = preprocessing.StandardScaler().fit(x_train)

x_train_norm = scaler.transform(x_train.astype(float))
x_train_norm[0:5]

x_test_norm = scaler.transform(x_test.astype(float))
x_test_norm[0:5]

k = 4

neigh = KNeighborsClassifier(n_neighbors = k).fit(x_train_norm, y_train)

print(y_test[0:10])
print(neigh.predict(x_test_norm[0:10]))

print("Train Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(x_train_norm)))
print("Test Accuracy: ", metrics.accuracy_score(y_test, neigh.predict(x_test_norm)))

# k = 6

# neigh = KNeighborsClassifier(n_neighbors = k).fit(x_train_norm, y_train)

# print("Train Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(x_train_norm)))
# print("Test Accuracy: ", metrics.accuracy_score(y_test, neigh.predict(x_test_norm)))

Ks = 10

mean_acc = np.zeros(Ks-1)
std_acc = np.zeros(Ks-1)

for i in range(1, Ks):

    neigh = KNeighborsClassifier(n_neighbors = i).fit(x_train_norm, y_train)

    y_hat = neigh.predict(x_test_norm)

    mean_acc[i-1] = metrics.accuracy_score(y_test, y_hat)

    std_acc[i-1]=np.std(y_hat==y_test)/np.sqrt(y_hat.shape[0])


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 