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

import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN
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

x = data['Close'].values
print(x)
xlen = x.shape[0]
data_train, data_test = x[0:int(0.7*xlen)], x[int(0.7*xlen+1):]

x_train = []
y_train = []
x_test = []
y_test = []

for i in range(60, data_train.shape[0]):
    x_train.append(data_train[i-60:i])
    y_train.append(data_train[i])

for i in range(60, data_test.shape[0]):
    x_test.append(data_test[i-60:i])
    y_test.append(data_test[i])

x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)
np.random.seed(5)
np.random.shuffle(x_train)
np.random.seed(5)
np.random.shuffle(x_test)
x_train = np.reshape(x_train, (x_train.shape[0], 60, 1))
x_test = np.reshape(x_test, (x_test.shape[0],60,1))
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(iii)
model = tf.kera


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

