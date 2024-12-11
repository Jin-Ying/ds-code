import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


df = pd.read_csv("aapl-trading-hour.csv")

null_num = df.isnull()
y = df["Close"]
x = df[["Close", "Open"]]

# we select independent variable

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.25,
                                                    shuffle = True,
                                                    random_state = 1)

print(x_train)