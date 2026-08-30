"""03-pandas / 08-cars-origin-type-mean：查看产地分布、品牌分布和车型均值。"""
import pandas as pd

cars = pd.read_csv('data/03-pandas/Cars93.csv')

print(cars.shape)
print()
print('产地分布:')
print(cars['Origin'].value_counts())
print()
print('品牌出现次数前 3:')
print(cars['Manufacturer'].value_counts().head(3))
print()

type_mean = cars.groupby('Type')[['Price', 'MPG.city', 'MPG.highway']].mean().round(1)
print(type_mean)
