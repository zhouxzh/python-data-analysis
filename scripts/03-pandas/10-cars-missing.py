"""03-pandas / 10-cars-missing：报告 Cars93 中的缺失列。"""
import pandas as pd

cars = pd.read_csv('data/03-pandas/Cars93.csv')

missing = cars.isna().sum()
print('有缺失的列:')
print(missing[missing > 0])
