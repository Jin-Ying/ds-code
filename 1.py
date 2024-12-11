import pandas as pd

import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, median_absolute_error
from timeseriesutil import TimeSeriesDiff, TimeSeriesEmbedder, ColumnExtractor

import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')
plt.xticks(rotation=70)

"""数据导入

这里我们导入经过了预处理的苹果公司股票报价，该数据仅包含正常交易时间的
数据。
"""

data = pd.read_csv("aapl-trading-hour.csv",
                   index_col=0)

y = data["Close"].diff() / data["Close"].shift()

y[np.isnan(y)] = 0

n_total = data.shape[0]
n_train = int(np.ceil(n_total * 0.7))

y_train = y[10:n_train]
y_test = y[(n_train + 10):]

data['MA_7MA'] = data["Close"].rolling(window=7).mean()
data['MA_15MA'] = data["Close"].rolling(window=15).mean()
ewma = pd.Series.ewm
data['close_30EMA'] = ewma(data["Close"], span=30).mean()

no_of_std = 2

data['MA_7MA'] = data["Close"].rolling(window=7).mean()
data['MA_7MA_std'] = data["Close"].rolling(window=7).std()
data['MA_7MA_BB_high'] = data['MA_7MA'] + no_of_std * data['MA_7MA_std']
data['MA_7MA_BB_low'] = data['MA_7MA'] - no_of_std * data['MA_7MA_std']

def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi

rsi_6 = rsiFunc(data['Close'].values, 6)

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].plot(data['MA_15MA'].values)
ax[1].plot(rsi_6)

data_train = data[:n_train]
data_test = data[n_train:]

""" 利用Pipeline实现建模
"""

pipeline = Pipeline([("ColumnEx", ColumnExtractor("Close")),
                     ("Diff", TimeSeriesDiff()),
                     ("Embed", TimeSeriesEmbedder(10)),
                     ("ImputerNA", SimpleImputer(missing_values=np.nan, strategy='mean')),
                     ("LinReg", LinearRegression())])

pipeline.fit(data_train, y_train)
y_pred = pipeline.predict(data_test)

""" 查看并评价结果
"""

print(r2_score(y_test, y_pred))
print(median_absolute_error(y_test, y_pred))

cc = np.sign(y_pred) * y_test
cumulative_return = (cc + 1).cumprod()
cumulative_return.plot(rot=10)
plt.savefig("plots/performance-simple-linreg.png")
# plt.show()
print(data)

"""更复杂的Pipeline

我们试图将成交量也纳入考虑，所以需要进行多个pipeline的融合。
同时，我们试图引入多远交互项，以考虑非线性相关关系。
"""

pipeline_closing_price = Pipeline([("ColumnEx", ColumnExtractor("Close")),
                                   ("Diff", TimeSeriesDiff()),
                                   ("Embed", TimeSeriesEmbedder(10)),
                                   ("ImputerNA", SimpleImputer(missing_values=np.nan, strategy='mean'))])

pipeline_volume = Pipeline([("ColumnEx", ColumnExtractor("MA_7MA")),
                            ("Diff", TimeSeriesDiff()),
                            ("Embed", TimeSeriesEmbedder(10)),
                            ("ImputerNA", SimpleImputer(missing_values=np.nan, strategy='mean'))])

merged_features = FeatureUnion([("ClosingPriceFeature", pipeline_closing_price),
                                ("VolumeFeature", pipeline_volume)])

pipeline_2 = Pipeline([("MergedFeatures", merged_features),
                       ("PolyFeature", PolynomialFeatures()),
                       ("LinReg", LinearRegression())])
pipeline_2.fit(data_train, y_train)

y_pred_2 = pipeline_2.predict(data_test)
print(y_test.shape)
print(y_pred_2.shape)
print(r2_score(y_test, y_pred_2))
print(median_absolute_error(y_test, y_pred_2))

cc_2 = np.sign(y_pred_2) * y_test
cumulative_return_2 = (cc_2 + 1).cumprod()
cumulative_return_2.plot(style="k--", rot=10)
plt.savefig("plots/performance-more-variables.png")
# plt.show()

""" 预测运行时间有多长?
"""

import time

start_time = time.clock()
pipeline_2.predict(data_test[1:20])
print(time.clock() - start_time, "seconds")

