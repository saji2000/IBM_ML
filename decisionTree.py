import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree

drugs = pd.read_csv('drug200.csv')

print(drugs.value_counts())