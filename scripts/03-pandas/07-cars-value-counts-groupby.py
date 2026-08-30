"""03-pandas / 07-cars-value-counts-groupby：查看车型分布并按车型汇总价格。"""
import pandas as pd

cars = pd.read_csv('data/03-pandas/Cars93.csv')

print(cars['Type'].value_counts())
print()
print(cars.groupby('Type')['Price'].agg(['count', 'mean']).round(1))
