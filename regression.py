import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

# data prep

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

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

# non-linear regression with degree 2

poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
print(train_x_poly)

train_y_ = regr.fit(train_x_poly, train_y)

print(regr.coef_)
print(regr.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')

XX = np.arange(0.0, 10.0, 0.1)

yy = regr.intercept_[0] + regr.coef_[0][1] *  XX + regr.coef_[0][2] * (XX ** 2)

plt.plot(XX, yy, '-r')

plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")

plt.show()

# non-linear regression testing with degree 2

test_x_poly = poly.transform(test_x)
test_y_ = regr.predict(test_x_poly)

print("Absolute Mean Error: %.2f" % np.mean(np.abs(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2 score: %.2f" % r2_score(test_y, test_y_))

# non-linear regression with degree 3

poly = PolynomialFeatures(degree=3)
train_x_poly = poly.fit_transform(train_x)

train_y_ = regr.fit(train_x_poly, train_y)

print(regr.intercept_)
print(regr.coef_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='black')

XX = np.arange(0.0, 10.0, 0.1)

yy = regr.intercept_[0] + regr.coef_[0][1] * XX + regr.coef_[0][2] * (XX ** 2) + regr.coef_[0][3] * (XX ** 3)

plt.plot(XX, yy, '-r')

plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")

plt.show()

# non-linear regression with degree 3

test_x_poly = poly.transform(test_x)
test_y_ = regr.predict(test_x_poly)

print("Mean absloute error: %.2f" % np.mean(np.abs(test_y_ - test_y)))
print("Residual Sum of Squares : %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2 score: %.2f" %  r2_score(test_y, test_y_))