import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn import preprocessing



drugs = pd.read_csv('drug200.csv')

x = drugs[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
x[:,1] = le_sex.transform(x[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['HIGH', 'LOW', 'NORMAL'])
x[:,2] = le_BP.transform(x[:,2])

le_Ch = preprocessing.LabelEncoder()
le_Ch.fit(['HIGH', 'LOW', 'NORMAL'])
x[:,3] = le_Ch.transform(x[:,3])

print(x)