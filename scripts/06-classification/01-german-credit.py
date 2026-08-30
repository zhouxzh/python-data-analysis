"""06-classification / 01-german-credit：用 LogisticRegression 做信贷风险分类。

运行：
    python scripts/06-classification/01-german-credit.py
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

german = pd.read_csv("data/07-modeling/GermanCredit.csv")
features = [
    "duration",
    "amount",
    "installment_rate",
    "present_residence",
    "age",
    "number_credits",
    "people_liable",
]
X = german[features]
y = german["credit_risk"]

print("标签分布:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_scaled, y_train)
pred = clf.predict(X_test_scaled)

print("accuracy:", round(accuracy_score(y_test, pred), 4))
print("confusion matrix:")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, zero_division=0))
