import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.1, 10.0, 0.1)

y = 1 + 4/(1 + np.power(x, 2 - 0.5))

y_noise = 1/4 * np.random.normal(size=x.size)

y_data = y + y_noise

plt.scatter(x, y_data, color='blue')
plt.plot(x, y, color='red')

plt.xlabel("Independet Variables")
plt.ylabel("Dependet Variables")

plt.show()

