"""07-regression / 03-boston-ridge：用 Ridge 正则化做对比。

运行：
    python scripts/07-regression/03-boston-ridge.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

boston = pd.read_csv("data/07-modeling/BostonHousing.csv")
X = boston.drop(columns=["medv"])
y = boston["medv"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train, y_train)
pred = ridge.predict(X_test)

print("ridge MAE:", round(mean_absolute_error(y_test, pred), 2))
print("ridge RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 2))
print("ridge R2:", round(r2_score(y_test, pred), 4))
