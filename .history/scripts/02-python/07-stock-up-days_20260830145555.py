"""02-python / 07-stock-up-days：数上涨天数。

先读取 Price 列，再逐日比较后一天是否高于前一天。
"""
import csv

path = "data/02-python/stock_price.csv"
prices = []

with open(path, encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        prices.append(float(row["Price"]))

up_days = 0
for i in range(1, len(prices)):
    if prices[i] > prices[i - 1]:
        up_days += 1

print("上涨天数:", up_days)
