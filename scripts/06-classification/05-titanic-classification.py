"""06-classification / 05-titanic-classification：Titanic 生存二分类练习。"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

titanic = pd.read_csv('data/07-modeling/titanic.csv')
print('shape:', titanic.shape)
print('survived 分布:')
print(titanic['survived'].value_counts())

features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
X = titanic[features].copy()
y = titanic['survived']

X['sex'] = X['sex'].map({'male': 0, 'female': 1})
X['embarked'] = X['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

imputer = SimpleImputer(strategy='median').fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

dummy = DummyClassifier(strategy='most_frequent', random_state=42).fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
print('dummy accuracy:', round(accuracy_score(y_test, dummy_pred), 4))

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
model.fit(X_train, y_train)
pred = model.predict(X_test)

print('logistic accuracy:', round(accuracy_score(y_test, pred), 4))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, digits=4))
