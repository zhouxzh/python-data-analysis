"""Week 7 实践：第一个预测模型。

运行：
    python scripts/week07_practice.py
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

print("=" * 60)
print("1. 读取并准备 Airbnb 数据")
print("=" * 60)

df = pd.read_csv("data/nyc_airbnb.csv")
df = df[(df["price"] > 0) & (df["price"] <= 1000)].copy()

features = [
    "room_type",
    "neighbourhood_group",
    "minimum_nights",
    "number_of_reviews",
    "availability_365",
    "calculated_host_listings_count",
    "latitude",
    "longitude",
]

print("样本数:", len(df))
print("缺失值:")
print(df[features].isna().sum())

print()
print("=" * 60)
print("2. 拆分并训练线性回归")
print("=" * 60)

X = pd.get_dummies(df[features], drop_first=True)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("train R2:", round(r2_score(y_train, train_pred), 4))
print("test R2:", round(r2_score(y_test, test_pred), 4))
print("test MAE:", round(mean_absolute_error(y_test, test_pred), 2))
print("测试集平均价格:", round(y_test.mean(), 2))

baseline = [y_test.mean()] * len(y_test)
print("baseline MAE（猜平均价）:", round(mean_absolute_error(y_test, baseline), 2))
print()
print("检查：是否先拆分再训练？是否报告 MAE 和基准？")
