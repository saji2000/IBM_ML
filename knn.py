import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df = pd.read_csv('teleCust1000t.csv')

x = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
x[0:5]

y = df['custcat'].values
y[0:5]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

print("Train set: ", x_train.shape, y_train.shape)

print("Test set: ", x_test.shape, y_test.shape)