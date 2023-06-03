import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit




df = pd.read_csv("china_gdp.csv")
df.head(10)

plt.figure(figsize=(8, 5))

x_data, y_data = (df['Year'].values, df['Value'].values)

plt.scatter(x_data, y_data, color='blue')
plt.xlabel('Year')
plt.ylabel('Value')
plt.show()

def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y

beta_1 = 0.10
beta_2 = 1990.0

#logistic function
Y_pred = sigmoid(x_data, beta_1 , beta_2)

#plot initial prediction against datapoints
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')

# plt.plot(x_data,Y_pred) 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

