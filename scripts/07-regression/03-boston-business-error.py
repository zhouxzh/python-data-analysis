"""07-regression / 03-boston-business-error：RMSE、误差分布和两个样本预测。"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('data/07-modeling/BostonHousing.csv')
features = ['crim', 'rm', 'lstat', 'ptratio', 'nox', 'age']
X = df[features]
y = df['medv']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression().fit(X_train, y_train)
test_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, test_pred) ** 0.5
resid = y_test - test_pred

print('测试集 RMSE：', round(rmse, 4))
print('绝对误差中位数：', round(resid.abs().median(), 4))
print('误差在 ±5 千美元内：', round((resid.abs() <= 5).mean(), 4))
print('误差在 ±10 千美元内：', round((resid.abs() <= 10).mean(), 4))

# 其他字段取全量中位数，只看 rm 和 lstat 变化
row_a = pd.DataFrame([{
    'crim': df['crim'].median(), 'rm': 6.0, 'lstat': 12.0,
    'ptratio': df['ptratio'].median(), 'nox': df['nox'].median(),
    'age': df['age'].median(),
}], columns=features)
row_b = pd.DataFrame([{
    'crim': df['crim'].median(), 'rm': 7.0, 'lstat': 5.0,
    'ptratio': df['ptratio'].median(), 'nox': df['nox'].median(),
    'age': df['age'].median(),
}], columns=features)

print('rm=6、lstat=12 的估计：', round(model.predict(row_a)[0], 4))
print('rm=7、lstat=5 的估计：', round(model.predict(row_b)[0], 4))
