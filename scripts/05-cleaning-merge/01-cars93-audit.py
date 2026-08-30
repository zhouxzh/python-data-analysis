"""05-cleaning-merge / 01-cars93-audit：缺失、重复和类型错误最小审计。"""
import pandas as pd

cars = pd.read_csv("data/04-cleaning/Cars93_miss.csv")
print("shape:", cars.shape)
print("重复行数:", int(cars.duplicated().sum()))
print("删除任意缺失值后的行数:", len(cars.dropna()))

cyl = pd.to_numeric(cars["Cylinders"], errors="coerce")
print("Cylinders 原始缺失:", int(cars["Cylinders"].isna().sum()))
print("Cylinders 强制转数值后缺失:", int(cyl.isna().sum()))
