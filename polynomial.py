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
# plt.show()

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
# plt.show()

# Lets normalize our data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))


# y = sigmoid(x, *popt)
# plt.plot(xdata, ydata, 'ro', label='data')
# plt.plot(x,y, linewidth=3.0, label='fit')
# plt.legend(loc='best')
# plt.ylabel('GDP')
# plt.xlabel('Year')
# plt.show()

x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
print(*popt)
y = sigmoid(x, *popt)

plt.scatter(xdata, ydata ,color='red', label='data')
plt.plot(x, y, linewidth=3.0,color='blue', label='fit')
plt.legend(loc='best')
plt.xlabel('Year')
plt.xlabel('GDP')
# plt.xlim(min(xdata), max(xdata))

# plt.show()