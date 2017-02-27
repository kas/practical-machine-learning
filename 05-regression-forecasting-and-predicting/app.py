import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL') # pandas dataframe
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True) # NaN data: in ML, we can't work with NaN data -- we have to replace NaN data with something -- will be treated as an outlier in our dataset

forecast_out = int(math.ceil(0.01 * len(df))) # number of days out
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1)) # features, features will be everything BUT the label column
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out] # last ~30 days

df.dropna(inplace=True)
y = np.array(df['label']) # labels
y = np.array(df['label']) # labels

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression() # define classifier
# clf = svm.SVR()
# clf = svm.SVR(kernel='poly')
# clf = LinearRegression(n_jobs=-1)

clf.fit(X_train, y_train) # fit/train classifier, we fit features and labels here, fit is synonymous with train
accuracy = clf.score(X_test, y_test) # could be named confidence instead, score is synonymous with test
# print(accuracy) # in linear regression, accuracy is squared error

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()