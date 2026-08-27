"""Week 7 实践：第一个预测模型。

运行：
    python scripts/week07_practice.py
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

print("=" * 60)
print("1. 读取 Titanic 数据")
print("=" * 60)

df = pd.read_csv("report/data/titanic.csv")
df = df[["survived", "pclass", "sex", "age", "fare"]].copy()
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["age"] = df["age"].fillna(df["age"].median())
df["fare"] = df["fare"].fillna(df["fare"].median())

print("shape:", df.shape)
print("缺失值:")
print(df.isna().sum())
print()
print("测试集生还比例提示：生还样本约占 38.75%")

print()
print("=" * 60)
print("2. 拆分并训练逻辑回归")
print("=" * 60)

X = df.drop(columns="survived")
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print("train accuracy:", round(train_acc, 4))
print("test accuracy:", round(test_acc, 4))
print()
print("confusion matrix:")
print(confusion_matrix(y_test, model.predict(X_test)))
print()
print(classification_report(y_test, model.predict(X_test), digits=4))

print()
print("检查：是否只用出发前可知字段？是否报告混淆矩阵和正类比例？")
