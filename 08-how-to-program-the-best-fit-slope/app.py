# m() = mean
# m = (m(x) * m(y) - m(x * y))/(m(x)^2 - m(x^2))

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

xs = np.array([1,2,3,4,5,6], dtype=np.float64) # float64 is the default dtype (data type)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def best_fit_slope(xs, ys):
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    mean_xy = np.mean(xs * ys)
    mean_xsquared = np.mean(np.square(xs))

    m = (((mean_x * mean_y) - mean_xy) / (np.square(mean_x) - mean_xsquared))

    return m

m = best_fit_slope(xs, ys)
print(m)

# demo
plt.scatter(xs, ys)
plt.plot(xs, (m * xs))
plt.show()
