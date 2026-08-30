"""07-regression / 01-boston-linear：线性回归、基线和 MAE/RMSE/R2。

运行：
    python scripts/07-regression/01-boston-linear.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

boston = pd.read_csv("data/07-modeling/BostonHousing.csv")
print("shape:", boston.shape)
print("目标列 medv 描述:")
print(boston["medv"].describe())

X = boston.drop(columns=["medv"])
y = boston["medv"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

baseline_pred = np.full(len(y_test), y_train.mean())
print("baseline MAE:", round(mean_absolute_error(y_test, baseline_pred), 2))
print("baseline RMSE:", round(np.sqrt(mean_squared_error(y_test, baseline_pred)), 2))
print("baseline R2:", round(r2_score(y_test, baseline_pred), 4))

reg = LinearRegression()
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
print("linear MAE:", round(mean_absolute_error(y_test, pred), 2))
print("linear RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 2))
print("linear R2:", round(r2_score(y_test, pred), 4))
