import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("china_gdp.csv")
df.head(10)

# plt.figure(figsize=(8, 5))

# x_data, y_data = (df['Year'].values, df['Value'].values)

# plt.scatter(x_data, y_data, color='blue')
# plt.xlabel('Year')
# plt.ylabel('Value')
# plt.show()
def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y

X = np.arange(-5.0, 5.0, 0.1)
Y = sigmoid(X, 0.1, 10)

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

