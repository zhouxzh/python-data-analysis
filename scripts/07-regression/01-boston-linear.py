"""07-regression / 01-boston-linear：线性回归、基线和 MAE/RMSE/R2。"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('data/07-modeling/BostonHousing.csv')
features = ['crim', 'rm', 'lstat', 'ptratio', 'nox', 'age']
X = df[features]
y = df['medv']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
baseline_pred = [y_test.mean()] * len(y_test)

print('样本量：全量', len(df), '，训练集', len(X_train), '，测试集', len(X_test))
print('训练集 R2：', round(r2_score(y_train, train_pred), 4))
print('测试集 R2：', round(r2_score(y_test, test_pred), 4))
print('测试集 MAE：', round(mean_absolute_error(y_test, test_pred), 4))
print('测试集 RMSE：', round(mean_squared_error(y_test, test_pred) ** 0.5, 4))
print('基线(猜均值) MAE：', round(mean_absolute_error(y_test, baseline_pred), 4))
print('基线(猜均值) RMSE：', round(mean_squared_error(y_test, baseline_pred) ** 0.5, 4))
print('测试集 medv 均值：', round(y_test.mean(), 4))
