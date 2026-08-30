"""06-classification / 01-german-credit-audit：核对 GermanCredit 标签含义。"""
import pandas as pd

german = pd.read_csv('data/07-modeling/GermanCredit.csv')
print(german.shape)
print(german['credit_risk'].value_counts())
print(pd.crosstab(german['credit_history'], german['credit_risk']))
