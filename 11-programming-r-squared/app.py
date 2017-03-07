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

def squared_error(ys_orig, ys_line):
    return sum(np.square(ys_line - ys_orig))

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = []
    for y in ys_orig:
        y_mean_line.append(mean(ys_orig))

    squared_error_regr = squared_error(ys_orig, ys_line)
    
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)

    return (1 - (squared_error_regr / squared_error_y_mean))

m, b = best_fit_slope_and_intercept(xs, ys)
print(m, b)

regression_line = []
for x in xs:
    regression_line.append((m * x) + b)

predict_x = 8
predict_y = ((m*predict_x) + b)

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='r')
plt.plot(xs, regression_line)
plt.show()

# demo
# plt.scatter(xs, ys)
# plt.plot(xs, (m * xs))
# plt.show()
