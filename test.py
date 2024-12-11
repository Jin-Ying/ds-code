import pandas as pd

import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, median_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler #for standardization
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import plot_confusion_matrix
from timeseriesutil import TimeSeriesDiff, TimeSeriesEmbedder, ColumnExtractor
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier, LGBMRegressor


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

data['MA_3MA'] = data["Close"].rolling(window=3).mean()
data['MA_5MA'] = data["Close"].rolling(window=5).mean()
data['MA_7MA'] = data["Close"].rolling(window=7).mean()
data['MA_15MA'] = data["Close"].rolling(window=15).mean()
data['MA_30MA'] = data["Close"].rolling(window=30).mean()

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
rsi_10 = rsiFunc(data['Close'].values, 10)
rsi_14 = rsiFunc(data['Close'].values, 14)

data['RSI_6'] = rsi_6
data['RSI_10'] = rsi_10
data['RSI_14'] = rsi_14


x = data[['Open', 'High', 'Low', 'Close', 'Volume','MA_3MA','MA_7MA','MA_15MA','MA_30MA','RSI_6','RSI_10','RSI_14']]


y = data["Close"].diff() / data["Close"].shift()
y[np.isnan(y)] = 0
x[np.isnan(x)] = 0
# y = np.where(y > 0, 1, 0)

x_tune = x
y_tune = y

x_train, x_test, y_train, y_test = train_test_split(x_tune, y_tune,
                                                    test_size = 0.25,
                                                    shuffle = True,
                                                    random_state = 1)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
y_pred= lr_model.predict(x_test)
r_sq = r2_score(y_test, y_pred)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {lr_model.intercept_}")
print(f"slope: {lr_model.coef_}")
print(iii)

lgbm = LGBMClassifier()

lgbm_params = {"n_estimators": [80, 100, 120, 150, 200],
              "max_depth": [-1, 3, 4],
              "learning_rate": [0.05, 0.08, 0.1, 0.12, 0.15, 0.18],
              "min_child_samples": [15, 20, 25, 30]}


lgbm_cv_model = GridSearchCV(lgbm, lgbm_params, cv = 10, n_jobs = -1)
lgbm_cv_model.fit(x_train, y_train)

print("Best score for train set: " + str(lgbm_cv_model.best_score_))

print("____________________________________________")

print("best learning_rate value: " + str(lgbm_cv_model.best_params_["learning_rate"]),
     "\nbest n_estimators value: " + str(lgbm_cv_model.best_params_["n_estimators"]),
     "\nbest max_depth value: " + str(lgbm_cv_model.best_params_["max_depth"]),
     "\nbest min_child_samples value: " + str(lgbm_cv_model.best_params_["min_child_samples"]))

lgbm = LGBMClassifier(learning_rate = lgbm_cv_model.best_params_["learning_rate"],
                      max_depth = lgbm_cv_model.best_params_["max_depth"],
                      n_estimators = lgbm_cv_model.best_params_["n_estimators"],
                      min_child_samples = lgbm_cv_model.best_params_["min_child_samples"])

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.25,
                                                    shuffle = True,
                                                    random_state = 1)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


lgbm_model = lgbm.fit(x_train, y_train)

y_pred = lgbm_model.predict(x_test)
print(y_pred)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

