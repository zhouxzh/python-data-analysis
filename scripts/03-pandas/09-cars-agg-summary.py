"""03-pandas / 09-cars-agg-summary：按车型聚合多个统计量并排序。"""
import pandas as pd

cars = pd.read_csv('data/03-pandas/Cars93.csv')

summary = cars.groupby('Type')[['Price', 'MPG.city']].agg(['count', 'mean', 'min', 'max']).round(1)
print(summary)
print()
print(summary.sort_values(('Price', 'mean'), ascending=False))
