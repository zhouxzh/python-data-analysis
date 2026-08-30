"""07-regression / 02-boston-coefficients：查看标准化系数方向。

运行：
    python scripts/07-regression/02-boston-coefficients.py
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

boston = pd.read_csv("data/07-modeling/BostonHousing.csv")
X = boston.drop(columns=["medv"])
y = boston["medv"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

reg = LinearRegression()
reg.fit(X_train, y_train)

coef = pd.Series(reg.coef_, index=X.columns).sort_values(key=abs, ascending=False)
print("绝对值最大的前 5 个系数:")
print(coef.head().round(3))
