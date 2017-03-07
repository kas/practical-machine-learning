# m() = mean
# m = (m(x) * m(y) - m(x * y))/(m(x)^2 - m(x^2))

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype=np.float64) # float64 is the default dtype (data type)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def best_fit_slope_and_intercept(xs, ys):
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    mean_xy = np.mean(xs * ys)
    mean_xsquared = np.mean(np.square(xs))

    m = (((mean_x * mean_y) - mean_xy) / (np.square(mean_x) - mean_xsquared))

    b = (mean_y - (m * mean_x))

    return m, b

m, b = best_fit_slope_and_intercept(xs, ys)
print(m, b)

regression_line = []
for x in xs:
    regression_line.append((m * x) + b)

predict_x = 8
predict_y = ((m*predict_x) + b)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='r')
plt.plot(xs, regression_line)
plt.show()

# demo
# plt.scatter(xs, ys)
# plt.plot(xs, (m * xs))
# plt.show()
