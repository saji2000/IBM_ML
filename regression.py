import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score


df = pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
df.head()

# summarize the data
df.describe()

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY','CO2EMISSIONS']]
cdf.head(9)

# viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
# viz.hist()
# plt.show()

# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

regr = linear_model.LinearRegression()
# x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
# y = np.asanyarray(train[['CO2EMISSIONS']])
# regr.fit (x, y)
# # The coefficients
# print ('Coefficients: ', regr.coef_)

# y_hat = regr.predict(test[['ENGINESIZE','CYLINDERS', 'FUELCONSUMPTION_COMB']])
# x = np.asanyarray(test[['ENGINESIZE','CYLINDERS', 'FUELCONSUMPTION_COMB']])
# y = np.asanyarray(test[['CO2EMISSIONS']])
# y
# # print("Residual sum of squares %.2f" % np.mean((y_hat - y) ** 2))
# print("Residual sum of squares %.2f" % np.mean((y_hat - y) ** 2))

# print('Variance: %.2f' % regr.score(x, y))

x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x, y)

print('Coefficients: ', regr.coef_)

y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
y = np.asanyarray(test[['CO2EMISSIONS']])

print('MSE: ', np.mean((y_hat - y) ** 2))

print('Variance: ', regr.score(x, y))