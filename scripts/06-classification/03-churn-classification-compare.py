"""06-classification / 03-churn-classification-compare：基线、默认模型和 balanced 模型对比。"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier

churn = pd.read_csv('data/07-modeling/Churn_Modelling.csv')
churn = churn.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

print(churn.shape)
print(churn['Exited'].value_counts(normalize=True).round(4).to_string())

X = pd.get_dummies(
    churn.drop(columns=['Exited']),
    columns=['Geography', 'Gender'],
    drop_first=True
)
y = churn['Exited']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 基线：永远预测多数类 0（不流失），不看任何特征
dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
print('dummy accuracy:', round(accuracy_score(y_test, dummy_pred), 4))

# 默认逻辑回归
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
pred = model.predict(X_test_scaled)

print('X shape:', X.shape)
print('train/test:', X_train.shape, X_test.shape)
print('accuracy:', round(accuracy_score(y_test, pred), 4))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, digits=4))

# 处理类别不平衡：给少数类更大权重
balanced = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
balanced.fit(X_train_scaled, y_train)
pred_bal = balanced.predict(X_test_scaled)

print('balanced accuracy:', round(accuracy_score(y_test, pred_bal), 4))
print(confusion_matrix(y_test, pred_bal))
print(classification_report(y_test, pred_bal, digits=4))
