"""06-classification / 04-churn-model-comparison：Dummy、LogisticRegression、DecisionTree 比较。"""
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline

churn = pd.read_csv('data/07-modeling/Churn_Modelling.csv')
churn = churn.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
X = pd.get_dummies(
    churn.drop(columns=['Exited']),
    columns=['Geography', 'Gender'],
    drop_first=True
)
y = churn['Exited']

models = {
    'Dummy': DummyClassifier(strategy='most_frequent', random_state=42),
    'LogisticRegression': make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    ),
    'DecisionTree': DecisionTreeClassifier(max_depth=4, random_state=42),
}

print('n_samples:', len(y))
for name, model in models.items():
    f1 = cross_val_score(model, X, y, cv=5, scoring='f1')
    auc = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print(name, 'f1_mean:', round(f1.mean(), 4), 'auc_mean:', round(auc.mean(), 4))
