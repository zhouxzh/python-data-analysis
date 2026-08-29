"""Week 7 实践：第一个预测模型。

运行：
    python scripts/week07_practice.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

print("=" * 60)
print("1. 信贷风险分类：GermanCredit.csv")
print("=" * 60)

german = pd.read_csv("data/07-modeling/GermanCredit.csv")
german_features = [
    "duration",
    "amount",
    "installment_rate",
    "present_residence",
    "age",
    "number_credits",
    "people_liable",
]
X = german[german_features]
y = german["credit_risk"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print("accuracy:", round(accuracy_score(y_test, pred), 4))
print("confusion matrix:")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, zero_division=0))

print()
print("=" * 60)
print("2. 波士顿房价回归：BostonHousing.csv")
print("=" * 60)

boston = pd.read_csv("data/07-modeling/BostonHousing.csv")
X = boston.drop(columns=["medv"])
y = boston["medv"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
reg = LinearRegression()
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
print("baseline MAE:", round(mean_absolute_error(y_test, [y_train.mean()] * len(y_test)), 2))
print("model MAE:", round(mean_absolute_error(y_test, pred), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 2))
print("R2:", round(r2_score(y_test, pred), 4))

print()
print("=" * 60)
print("3. 银行客户流失分类：Churn_Modelling.csv")
print("=" * 60)

churn = pd.read_csv("data/07-modeling/Churn_Modelling.csv")
churn_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
X = churn[churn_features]
y = churn["Exited"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print("accuracy:", round(accuracy_score(y_test, pred), 4))
print("confusion matrix:")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
