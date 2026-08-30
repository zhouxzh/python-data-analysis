"""07-regression / 02-boston-coefficients：标准化系数、相关性和 Ridge 对比。"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('data/07-modeling/BostonHousing.csv')
features = ['crim', 'rm', 'lstat', 'ptratio', 'nox', 'age']
X = df[features]
y = df['medv']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 系数方向：先标准化，让不同量纲的系数可以比较大小
pipe = make_pipeline(StandardScaler(), LinearRegression())
pipe.fit(X_train, y_train)
coef = pipe.named_steps['linearregression'].coef_

print('标准化后的系数（按绝对值从大到小）：')
for name, c in sorted(zip(features, coef), key=lambda t: -abs(t[1])):
    print(' ', name, round(c, 4))

print('各特征与 medv 的相关系数：')
for name in features:
    print(' ', name, round(df[name].corr(df['medv']), 4))

print('rm 与 lstat 相关系数：', round(df['rm'].corr(df['lstat']), 4))
print('nox 与 age 相关系数：', round(df['nox'].corr(df['age']), 4))

# Ridge 正则化对比
lin = LinearRegression().fit(X_train, y_train)
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_train, y_train)

print('线性回归 测试 R2：', round(r2_score(y_test, lin.predict(X_test)), 4))
print('Ridge    测试 R2：', round(r2_score(y_test, ridge.predict(X_test)), 4))
print('线性回归 测试 RMSE：', round(mean_squared_error(y_test, lin.predict(X_test)) ** 0.5, 4))
print('Ridge    测试 RMSE：', round(mean_squared_error(y_test, ridge.predict(X_test)) ** 0.5, 4))
