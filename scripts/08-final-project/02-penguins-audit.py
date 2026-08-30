"""08-final-project / 02-penguins-audit：企鹅数据只读审计。"""
import pandas as pd

penguins = pd.read_csv('data/08-final/penguins.csv')
print('shape:', penguins.shape)
print('dtypes:')
print(penguins.dtypes)
print('缺失值:')
print(penguins.isna().sum())
print('species 计数:')
print(penguins['species'].value_counts())
