import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import sklearn.tree as tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import graphviz


drugs = pd.read_csv('drug200.csv')

x = drugs[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

y = drugs['Drug']


# converting categorical type of data into int because sklearn does not support them
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
x[:,1] = le_sex.transform(x[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['HIGH', 'LOW', 'NORMAL'])
x[:,2] = le_BP.transform(x[:,2])

le_Ch = preprocessing.LabelEncoder()
le_Ch.fit(['HIGH', 'LOW', 'NORMAL'])
x[:,3] = le_Ch.transform(x[:,3])

# train/test split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)

# decision tree

drugTree = DecisionTreeClassifier(criterion='entropy', max_depth=4)

drugTree.fit(x_train, y_train)

# testing

predTree = drugTree.predict(x_test)

print(predTree[0:5])
print(y_test[0:5])

print("Decision Tree's Accuracy: ", metrics.accuracy_score(y_test, predTree))

# Visualize decision tree

tree.plot_tree(drugTree)
plt.show()

# dot_data = export_graphviz(drugTree, feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'],
#                            class_names=drugTree.classes_, filled=True, rounded=True)
# graph = graphviz.Source(dot_data)
# graph.render('drug_tree', format='png', cleanup=True)
# graph.view()