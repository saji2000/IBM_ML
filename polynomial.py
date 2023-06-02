import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("china_gdp.csv")
df.head(10)

plt.figure(figsize=(8, 5))

x_data, y_data = (df['Year'].values, df['Value'].values)

plt.scatter(x_data, y_data, color='blue')
plt.xlabel('Year')
plt.ylabel('Value')
plt.show()



