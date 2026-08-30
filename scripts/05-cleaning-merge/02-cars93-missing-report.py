"""05-cleaning-merge / 02-cars93-missing-report：完整报告 Cars93_miss 的缺失和删除影响。"""
import pandas as pd

cars = pd.read_csv("data/04-cleaning/Cars93_miss.csv")
print("shape:", cars.shape)
print("重复行数:", int(cars.duplicated().sum()))
missing = cars.isna().sum()
print("每列缺失数:")
print(missing[missing > 0])
print("删除任意缺失值后的 shape:", cars.dropna().shape)
