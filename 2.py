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
from lightgbm import LGBMRegressor


matplotlib.style.use('ggplot')
plt.xticks(rotation=70)

"""数据导入

这里我们导入经过了预处理的苹果公司股票报价，该数据仅包含正常交易时间的
数据。
"""

data = pd.read_csv("aapl-trading-hour.csv",
                   index_col=0)

n_total = data.shape[0]
n_train = int(np.ceil(n_total * 0.7))

data_train = data[:n_train]
data_test = data[n_train:]

from sklearn.model_selection import train_test_split
market_train_df = data_train

market_train_df = market_train_df.reset_index()
market_train_df = market_train_df.drop(columns='index')

# Random train-test split
train_indices, val_indices = train_test_split(market_train_df.index.values,test_size=0.1, random_state=92)

feature_columns = ['Open','Volume']

def get_input(market_train, indices):
    X = market_train.loc[indices, feature_columns].values
    return X

y = market_train_df["Close"].diff() / market_train_df["Close"].shift()

X_train = get_input(market_train_df, train_indices)
X_val = get_input(market_train_df, val_indices)
y_train = y.loc[train_indices].diff()/y.loc[train_indices].shift()
y_val = y.loc[val_indices].diff()/y.loc[val_indices].shift()
y_train[np.isnan(y_train)] = 0
y_val[np.isnan(y_val)] = 0
y_train[np.isinf(y_train)] = 0
y_val[np.isinf(y_val)] = 0

def learning_rate_power(current_round):
    base_learning_rate = 0.19000424246380565
    min_learning_rate = 0.01
    lr = base_learning_rate * np.power(0.995,current_round)
    return max(lr, min_learning_rate)


gbm = LGBMRegressor(objective='regression', num_leaves=310, learning_rate=0.05, n_estimators=50)
gbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='l2', early_stopping_rounds=100)
