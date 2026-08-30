"""06-classification / 02-german-credit-logistic：GermanCredit LogisticRegression 建模。"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

german = pd.read_csv('data/07-modeling/GermanCredit.csv')

cat_cols = german.drop(columns=['credit_risk']).select_dtypes(include='object').columns.tolist()
X = pd.get_dummies(german.drop(columns=['credit_risk']), columns=cat_cols, drop_first=True)
y = german['credit_risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
pred = model.predict(X_test_scaled)

print('X shape:', X.shape)
print('train/test:', X_train.shape, X_test.shape)
print('accuracy:', round(accuracy_score(y_test, pred), 4))
print('confusion matrix (rows=actual, cols=predicted):')
print(confusion_matrix(y_test, pred))
