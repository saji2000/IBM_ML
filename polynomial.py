import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5.0, 5.0, 0.1)

y = 2 + 1.5 * (3 ** x)

y_noise = 5 * np.random.normal(size=x.size)

y_data = y + y_noise

plt.scatter(x, y_data, color='blue')
plt.plot(x, y, color='red')

plt.xlabel("Independet Variables")
plt.ylabel("Dependet Variables")

plt.show()

