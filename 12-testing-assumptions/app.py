# m() = mean
# m = (m(x) * m(y) - m(x * y))/(m(x)^2 - m(x^2))

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

# xs = np.array([1,2,3,4,5,6], dtype=np.float64) # float64 is the default dtype (data type)
# ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    
    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope_and_intercept(xs, ys):
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    mean_xy = np.mean(xs * ys)
    mean_xsquared = np.mean(np.square(xs))

    m = (((mean_x * mean_y) - mean_xy) / (np.square(mean_x) - mean_xsquared))

    b = (mean_y - (m * mean_x))

    return m, b

def squared_error(ys_orig, ys_line):
    return sum(np.square(ys_line - ys_orig))

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = []
    for y in ys_orig:
        y_mean_line.append(mean(ys_orig))

    squared_error_regr = squared_error(ys_orig, ys_line)
    
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)

    return (1 - (squared_error_regr / squared_error_y_mean))

xs, ys = create_dataset(40, 80, 2, correlation=False)

m, b = best_fit_slope_and_intercept(xs, ys)
# print(m, b)

regression_line = []
for x in xs:
    regression_line.append((m * x) + b)

predict_x = 8
predict_y = ((m*predict_x) + b)

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.scatter(predict_x, predict_y, s=100, color='r')
plt.show()

# demo
# plt.scatter(xs, ys)
# plt.plot(xs, (m * xs))
# plt.show()
