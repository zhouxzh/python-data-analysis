"""05-cleaning-merge / 01-cars93-missing：缺失、重复和删除缺失行的影响。

运行：
    python scripts/05-cleaning-merge/01-cars93-missing.py
"""
import pandas as pd

cars = pd.read_csv("data/04-cleaning/Cars93_miss.csv")
print("shape:", cars.shape)
print("重复行数:", int(cars.duplicated().sum()))
missing = cars.isna().sum()
print("每列缺失数:")
print(missing[missing > 0])
print("删除任意缺失值后的 shape:", cars.dropna().shape)
